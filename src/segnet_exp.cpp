#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <time.h>  

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "segnet.h"
#include "densecrf/densecrf.h"
#include "td.h"

using namespace std;
using namespace cv;

#define NODE "idr_segimg_node"
#define DEBUGDISPLAY 1

Mat img(cvSize(480,360),CV_8UC3,Scalar(0,0,0));

/*Dense CRF necessary function*/
const float GT_PROB = 0.5;
// Simple classifier that is 50% certain that the annotation is correct
MatrixXf computeUnary( const VectorXs & lbl, int M ){
	const float u_energy = -log( 1.0 / M );
	const float n_energy = -log( (1.0 - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	MatrixXf r( M, lbl.rows() );
	r.fill(u_energy);
	//printf("%d %d %d \n",im[0],im[1],im[2]);
	for( int k=0; k<lbl.rows(); k++ ){
		// Set the energy
		if (lbl[k]>=0){
			r.col(k).fill( n_energy );
			r(lbl[k],k) = p_energy;
		}
	}
	return r;
}

/*crop image*/
int imageCrop(InputArray src, OutputArray dst, Rect rect)  
{  
    Mat input = src.getMat();  
    if( input.empty() )
    {  
        return -1;  
    }  
  
    //计算剪切区域：  剪切Rect与源图像所在Rect的交集  
    Rect srcRect(0, 0, input.cols, input.rows);  
    rect = rect & srcRect;  
    if ( rect.width <= 0  || rect.height <= 0 )
    	return -2;  
  
    //创建结果图像  
    dst.create(Size(rect.width, rect.height), src.type());  
    Mat output = dst.getMat();  
    if ( output.empty() )
    	return -1;  
  
    try {  
        //复制源图像的剪切区域 到结果图像  
        input(rect).copyTo( output );  
        return 0;  
    } catch (...) {  
        return -3;  
    }  
}

/*recieve img msg callback function*/
/*void img_sub_callback(const sensor_msgs::ImageConstPtr &msg)
{
	cv_bridge::CvImagePtr cv_ptr;
    cv_ptr =  cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
	img = cv_ptr->image;
}*/

int main(int argc, char** argv) 
{
	ros::init(argc, argv, NODE);
	ros::NodeHandle node;

	// Set up publishers
	image_transport::ImageTransport it(node);
	image_transport::Publisher im_publisher;
	im_publisher = it.advertise("seg_res",1);

	// Set up subscribers //to be revised
	//ros::Subscriber sub;
    //sub = node.subscribe("/dirvingimgs",1000,img_sub_callback);

	// Load network
	Classifier classifier;
	string colorfile = "/home/inin/catkin_ws/src/segnetweight/color.png";
	Mat color = imread(colorfile, 1);

	while(ros::ok())
	{
		// Load image
		char file_name[256];

		//sprintf(file_name, "/home/inin/slam3d/dataset/KITTI_SEMANTIC/Training_00/RGB/1/%06d.png", i);
		sprintf(file_name, "/home/inin/catkin_ws/src/idr_segimg/data/3img.bmp");
		
		////* frame: origin size  temp_frame: 480*360 *////
		Mat frame = imread(file_name, 1);

		Mat temp_frame;
		resize(frame, temp_frame, Size(480,360));
		if(temp_frame.size().width <= 0)
			continue;

#if DEBUGDISPLAYW
		imshow("resize frame image",temp_frame);
#endif

		// time
		clock_t starttime=clock();

		//////////////////////////////////////////////////

		/////////////// segnet process ///////////////////

		// Prediction		
		vector<Prediction> predictions = classifier.Classify(temp_frame);

		Mat segnet(temp_frame.size(), CV_8UC3, Scalar(0,0,0));
		for (int i = 0; i < 360; ++i)
		{	
			uchar* segnet_ptr = segnet.ptr<uchar>(i);
			for (int j = 0; j < 480; ++j)
			{
				segnet_ptr[j*3+0] = predictions[i*480+j].second;
				segnet_ptr[j*3+1] = predictions[i*480+j].second;
				segnet_ptr[j*3+2] = predictions[i*480+j].second;
			}
		}

#if DEBUGDISPLAY
		imshow("segnet raw",segnet);
#endif

		// recover
		resize(segnet, segnet, frame.size());

		LUT(segnet, color, segnet);

#if DEBUGDISPLAY
		imshow("segnet", segnet);
#endif

		clock_t segtime=clock();
		cout<<"*************"<<endl;
		cout<<"segtime: "<<(segtime - starttime)/1000<<" ms"<<endl;

		//////////////////////////////////////////////////

		/////////////// dense crf ////////////////////////

	    const int M = 12; // Number of labels

	    int W  = frame.cols;
	    int H  = frame.rows;
	    int GW = segnet.cols;
	    int GH = segnet.rows;

		DenseCRF2D crf(W, H, M);
		
		/*prepare images as uchar ptr format*/
	    unsigned char * im;
		im = crf.convert2ppm(frame);

		unsigned char * anno;
		anno = crf.convert2ppm(segnet);

		crf.nColors=0;
		MatrixXf unary = computeUnary( crf.getLabeling( anno, W*H, M, crf.colors, crf.nColors), M );

		crf.setUnaryEnergy( unary );

		crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 3 ) );

		crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new PottsCompatibility( 10 ) );

	    VectorXs map = crf.map(5);

	    unsigned char *res = crf.colorize( map, W, H, crf.colors);

	    Mat crfResultImg( H, W, CV_8UC3, Scalar(0,0,0) );
	    crfResultImg = crf.ppm2mat(res, W, H);

#if DEBUGDISPLAY
		imshow("crfResultImg",crfResultImg);
#endif

	    // Counting time
		clock_t dcrftime=clock();
		cout<<"DCRF time: "<<(dcrftime - segtime)/1000<<" ms"<<endl;
		
		//////////////////////////////////////////////////
		///////// extract traversable directions /////////

	    Mat cropedCrfResultImg;
	    imageCrop(crfResultImg, cropedCrfResultImg, Rect(0,180,480,360));

#if DEBUGDISPLAY
	    imshow("cropedCrfResultImg", cropedCrfResultImg);
	    //imwrite("cropedCrfResultImg.png",cropedCrfResultImg);
#endif

	    TD_detector td(cropedCrfResultImg.cols, cropedCrfResultImg.rows);

	    td.calMaxDistance();
	    
	    td.getNGpoints(cropedCrfResultImg);

	    //td.drawGraph(360,td.TravDir);
	    clock_t endtime2=clock();
	    std::cout<<"getNGpoints time: "<<(endtime2 - dcrftime)/1000<<" ms"<<endl;

		waitKey(0);
	}

	ros::shutdown();
	return 0;
}