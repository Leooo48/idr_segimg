#include "common_headers.h"
#include "segnet.h"
#include "densecrf.h"
#include "td.h"

using namespace std;
using namespace cv;

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

int main()               
{
	// Load segnet network
	Classifier classifier;
	string colorfile = "/home/leo/cross_leo/model/color.png";
	cv::Mat color = cv::imread(colorfile, 1);

	int img_num = 1;
	for (int i = 0; i < img_num; i++)
	{
		char file_name[256];
		//sprintf(file_name, "/home/leo/slam3d/dataset/KITTI_SEMANTIC/Training_00/RGB/1/%06d.png", i);
		sprintf(file_name, "/home/leo/cross_leo/data/2img.bmp");

		cv::Mat frame = cv::imread(file_name, 1);
		cv::Mat temp_frame;
		cv::resize(frame, temp_frame, cv::Size(480,360));
		if(temp_frame.size().width <= 0)continue;
		
		// time
		clock_t starttime=clock();

		std::vector<Prediction> predictions = classifier.Classify(temp_frame);
		cv::Mat segnet(temp_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
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
		cv::resize(segnet, segnet, frame.size());
		cv::LUT(segnet, color, segnet);
		imshow("segnet",segnet);
		imwrite("segnet.png",segnet);
	    //waitKey(0);

	    clock_t segtime = clock();

		// Number of labels
	    const int M = 12;
	    int W  = frame.cols;
	    int H  = frame.rows;
	    int GW = segnet.cols;
	    int GH = segnet.rows;
		DenseCRF2D crf(W, H, M);

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
	    cv::Mat res_png(H,W,CV_8UC3,Scalar(0,0,0));
	    res_png = crf.ppm2mat(res, W, H);

	    // Counting time
		clock_t endtime=clock();
		std::cout<<"Seg time: "<<(segtime - starttime)/1000<<" ms"<<endl;
		std::cout<<"DCRF time: "<<(endtime - segtime)/1000<<" ms"<<endl;

	    delete[] im;
		delete[] anno;
		delete[] res;

	    imshow("result",res_png);
	    imwrite("result.png",res_png);

	    /*extract traversable directions*/
	    Mat cropres_png;
	    imageCrop(res_png,cropres_png,Rect(0,180,480,360));
	    imshow("crop",cropres_png);
	    imwrite("crop.png",cropres_png);

	    TD_detector td(cropres_png.cols, cropres_png.rows);
	    td.calMaxDistance();
	    td.getNGpoints(cropres_png);
	    td.drawGraph(360,td.TravDir);
	    clock_t endtime2=clock();
	    std::cout<<"getNGpoints time: "<<(endtime2 - endtime)/1000<<" ms"<<endl;
	}

    return 0;
}
