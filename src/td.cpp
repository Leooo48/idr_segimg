#include "td.h"
#include <iostream>

using namespace cv;
using namespace std;

#define PI 3.141592653

/*some useful tool function*/
void linearSmooth( double in[], double out[], int N )
{
    int i;
    if ( N < 5 )
    {
        for ( i = 0; i <= N - 1; i++ )
        {
            out[i] = in[i];
        }
    }
    else
    {
        out[0] = ( 3.0 * in[0] + 2.0 * in[1] + in[2] - in[4] ) / 5.0;
        out[1] = ( 4.0 * in[0] + 3.0 * in[1] + 2 * in[2] + in[3] ) / 10.0;
        for ( i = 2; i <= N - 3; i++ )
        {
            out[i] = ( in[i - 2] + in[i - 1] + in[i] + in[i + 1] + in[i + 2] ) / 5.0;
        }
        out[N - 2] = ( 4.0 * in[N - 1] + 3.0 * in[N - 2] + 2 * in[N - 3] + in[N - 4] ) / 10.0;
        out[N - 1] = ( 3.0 * in[N - 1] + 2.0 * in[N - 2] + in[N - 3] - in[N - 5] ) / 5.0;
    }
}

void findPeak(const vector<int>& v,  vector<int>& peakPositions)
{
    vector<int> diff_v (v.size(), 0);
    
    for (vector<int>::size_type i = 0; i !=diff_v.size(); i++)
    {
        if (v[i + 1] - v[i]>0)
            diff_v[i] = 1;
        else if (v[i + 1] - v[i] < 0)
            diff_v[i] = -1;
        else
            diff_v[i] = 0;
    }
    diff_v[v.size()-1] = 
        v[0]-v[v.size()-1]>0 ? 1 : ( v[0]-v[v.size()-1]<0 ? -1 : 0 );
    
    for (int i = diff_v.size() - 1; i >= 0; i--)
    {
        if (diff_v[i] == 0 && i == diff_v.size()-1)
        {
            diff_v[i] = 1;
        }
        else if (diff_v[i] == 0)
        {
            if (diff_v[i + 1] >= 0)
                diff_v[i] = 1;
            else
                diff_v[i] = -1;
        }
    }

    for (vector<int>::size_type i = 0; i != diff_v.size() - 1; i++)
    {
        if (diff_v[i + 1] - diff_v[i] > 0 )//peak:<0 hollow:>0
            peakPositions.push_back(i + 1);
    }
}

typedef pair<int, double> P;
bool cmp(const P &a, const P &b)
{
    if (a.second < b.second)
        return true;
    else return false;
}

int icvprCcaBySeedFill(const Mat& _binImg, Mat& _lableImg)  
{  
    // connected component analysis (4-component)  
    // use seed filling algorithm  
    // 1. begin with a foreground pixel and push its foreground neighbors into a stack;  
    // 2. pop the top pixel on the stack and label it with the same label until the stack is empty  
    //   
    // foreground pixel: _binImg(x,y) = 1  
    // background pixel: _binImg(x,y) = 0  
  
  
    if (_binImg.empty() ||  
        _binImg.type() != CV_8UC1)  
    {  
        return 0;  
    }  
  
    _lableImg.release() ;  
    _binImg.convertTo(_lableImg, CV_32SC1) ;  
  
    int label = 1 ;  // start by 2  
  
    int rows = _binImg.rows - 1 ;  
    int cols = _binImg.cols - 1 ;  
    for (int i = 1; i < rows-1; i++)  
    {  
        int* data= _lableImg.ptr<int>(i) ;  
        for (int j = 1; j < cols-1; j++)  
        {  
            if (data[j] == 1)  
            {  
                stack<pair<int,int>> neighborPixels ;     
                neighborPixels.push(pair<int,int>(i,j)) ;     // pixel position: <i,j>  
                ++label ;  // begin with a new label  
                while (!neighborPixels.empty())  
                {  
                    // get the top pixel on the stack and label it with the same label  
                    pair<int,int> curPixel = neighborPixels.top() ;  
                    int curX = curPixel.first ;  
                    int curY = curPixel.second ;  
                    _lableImg.at<int>(curX, curY) = label ;  
  
                    // pop the top pixel  
                    neighborPixels.pop() ;  
  
                    // push the 4-neighbors (foreground pixels)  
                    if (_lableImg.at<int>(curX, curY-1) == 1)  
                    {// left pixel  
                        neighborPixels.push(pair<int,int>(curX, curY-1)) ;  
                    }  
                    if (_lableImg.at<int>(curX, curY+1) == 1)  
                    {// right pixel  
                        neighborPixels.push(pair<int,int>(curX, curY+1)) ;  
                    }  
                    if (_lableImg.at<int>(curX-1, curY) == 1)  
                    {// up pixel  
                        neighborPixels.push(pair<int,int>(curX-1, curY)) ;  
                    }  
                    if (_lableImg.at<int>(curX+1, curY) == 1)  
                    {// down pixel  
                        neighborPixels.push(pair<int,int>(curX+1, curY)) ;  
                    }  
                }         
            }  
        }  
    } 

    return label;
}

Scalar icvprGetRandomColor()  
{  
    uchar r = 255 * (rand()/(1.0 + RAND_MAX));  
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));  
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));  
    return Scalar(b,g,r) ;  
}  
  
void icvprLabelColor(const Mat& _labelImg, Mat& _colorLabelImg)   
{  
    if (_labelImg.empty() ||  
        _labelImg.type() != CV_32SC1)  
    {  
        return ;  
    }  
  
    map<int, Scalar> colors ;  
  
    int rows = _labelImg.rows ;  
    int cols = _labelImg.cols ;  
  
    _colorLabelImg.release() ;  
    _colorLabelImg.create(rows, cols, CV_8UC3) ;  
    _colorLabelImg = Scalar::all(0) ;  
  
    for (int i = 0; i < rows; i++)  
    {  
        const int* data_src = (int*)_labelImg.ptr<int>(i) ;  
        uchar* data_dst = _colorLabelImg.ptr<uchar>(i) ;  
        for (int j = 0; j < cols; j++)  
        {  
            int pixelValue = data_src[j] ;  
            if (pixelValue > 1)  
            {  
                if (colors.count(pixelValue) <= 0)  
                {  
                    colors[pixelValue] = icvprGetRandomColor() ;  
                }  
                Scalar color = colors[pixelValue] ;  
                *data_dst++   = color[0] ;  
                *data_dst++ = color[1] ;  
                *data_dst++ = color[2] ;  
            }  
            else  
            {  
                data_dst++ ;  
                data_dst++ ;  
                data_dst++ ;  
            }  
        }  
    }  
}

void IPMprocess(const Mat& _src, Mat& _dst)
{

	Point2f p1,p2,p3,p4,p1r,p2r,p3r,p4r;
	vector<Point2f> capturePoint_R,corners_R;

	Mat tansfor_pR;
	double range=24.0,gridsize=0.04;

	p1 = Point2f(((double)range - 0.5) / gridsize, (2.0 * (double)range - 3.4) / gridsize);
	p2 = Point2f(((double)range - 0.5) / gridsize, (2.0 * (double)range - 4.4) / gridsize);
	p3 = Point2f(((double)range + 0.5) / gridsize, (2.0 * (double)range - 4.4) / gridsize);
	p4 = Point2f(((double)range + 0.5) / gridsize, (2.0 * (double)range - 3.4) / gridsize);

	corners_R.push_back(p1);
	corners_R.push_back(p2);
	corners_R.push_back(p3);
	corners_R.push_back(p4);

	p1r = Point2f(232,112);
	p2r = Point2f(234,101);
	p3r = Point2f(277,101);
	p4r = Point2f(280,112);

	capturePoint_R.push_back(p1r);
	capturePoint_R.push_back(p2r);
	capturePoint_R.push_back(p3r);
	capturePoint_R.push_back(p4r);

	tansfor_pR = getPerspectiveTransform(capturePoint_R, corners_R);

	Mat ipm_img;
	warpPerspective(_src, _dst, tansfor_pR, Size(2*range/gridsize,2*range/gridsize));
	resize(_dst,_dst,Size(400,400));

	for(int i=0;i<_dst.rows;i++)
		for(int j=0;j<_dst.cols;j++)
			_dst.at<unsigned char>(i,j)=
			    (_dst.at<unsigned char>(i,j)==0)?0:255;

}

void findTopo(cv::Mat& disTF, cv::Point _maxLoc)
{
	int dir[7][360];

    for(int i=0;i<7;i++)
		for(int j=0;i<360;i++)
			dir[i][j]=0;

    for(int i=0;i<disTF.rows;i++)  
    {  
        for(int j=0;j<disTF.cols;j++)  
        {
        	if(disTF.at<unsigned char>(i,j)>0)
        	{
        		int dis = (int)sqrt((i-_maxLoc.y)*(i-_maxLoc.y)+(j-_maxLoc.x)*(j-_maxLoc.x));
        		int n = dis / 10;

        		if(dis%10==0 && n<=7 && n>0)
        		{
        			int gamma = (int)(180 + atan2(_maxLoc.y - i, j - _maxLoc.x) / PI*180.0);

					if(gamma == 360)
						gamma = 0;
					if(gamma >=0 && gamma<=180)
						gamma = 180 - gamma;
					else if(gamma > 180 && gamma < 360)
						gamma = 540 - gamma;

					dir[n][gamma] = 
					    disTF.at<unsigned char>(i,j)>dir[n][gamma]?
					    disTF.at<unsigned char>(i,j):dir[n][gamma];
        		}
        	}
        }
	}

	vector<int> v[7];
	for(int i=0;i<7;i++)
	{
		vector<int> vecAngle(dir[i], dir[i]+sizeof(dir[i])/sizeof(int)); 
		vector<int> peakPositions;
		findPeak(vecAngle,peakPositions);
		v[i]=peakPositions;
	}

	/*for(int i=0;i<7;i++)
	{
		cout<<"v["<<i<<"] --> ";
		for(int j=0;j<v[i].size();j++)
			cout<<v[i][j]<<", ";
		cout<<endl;
	}*/

	//imshow("dont know",disTF);
}

void TD_detector::getNGpoints(Mat src)
{
	/*1. show road part*/
	Mat Road(src.size(),CV_8UC1,Scalar(0));
	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			/*Road:128 64 128 Mark:0 69 255*/
			if (   (src.at<Vec3b>(i, j)[0] == 128  &&
			        src.at<Vec3b>(i, j)[1] == 64   &&
			        src.at<Vec3b>(i, j)[2] == 128) ||
				   (src.at<Vec3b>(i, j)[0] == 0    &&
			        src.at<Vec3b>(i, j)[1] == 69   &&
			        src.at<Vec3b>(i, j)[2] == 255) )
				Road.at<unsigned char>(i, j)=1;
		}
	}

	/*connected conponent analysis*/
	Mat labelImg1,labelImg2,colorImg;

    int Label1=icvprCcaBySeedFill(Road, labelImg1);
    labelImg1.convertTo(labelImg1, CV_8UC1); 

    unsigned char road_pixel = labelImg1.at<unsigned char>(Road.rows-1, Road.cols/2);
    for(int i=0;i<Road.rows;i++)
		for(int j=0;j<Road.cols;j++)
			Road.at<unsigned char>(i, j)=
			  labelImg1.at<unsigned char>(i, j)==road_pixel?0:1;
	
    int Label2=icvprCcaBySeedFill(Road, labelImg2);
    //icvprLabelColor(labelImg2,colorImg);
    
    /*calculate point numbers in each connected conponent*/
	int *numLabel = new int[Label2+1];
	vector<Point> *pointsLabel = new vector<Point>[Label2+1];

	for (int i = 0; i < Label2+1; i++)
		numLabel[i] = 0;

	for (int i = 0; i < labelImg2.rows; i++)
	{
		for (int j = 0; j < labelImg2.cols; j++)
		{
			numLabel[labelImg2.at<int>(i, j)]++;
			pointsLabel[labelImg2.at<int>(i, j)].push_back(Point(j, i));
		}
	}
	
	/*remove small blocks*/
	for (int i = 2; i < Label2+1; i++)
		if (numLabel[i]<20)
			for(int j=0; j<numLabel[i]; j++)
				Road.at<unsigned char>(pointsLabel[i][j].y, pointsLabel[i][j].x)=0;

	for(int i=0; i<Road.rows; i++)
		for(int j=0; j<Road.cols; j++)
			Road.at<unsigned char>(i, j)=
		        ( Road.at<unsigned char>(i, j)==0?255:0 );

	imshow("Road",Road);
	//imwrite("Road.png",Road);
	delete []numLabel;

	/*2. calculate distances of non-ground points*/
	for(int i=0;i<360;i++)
    {
    	Angle[i]=DMax[i];
    	//Angle[i]=0;
    }

	double theta_d = 0;
	int theta_z;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			if (Road.at<unsigned char>(j, i)!=255)
			{
				theta_d = atan2(h/2 - j, i - w/2);

				theta_z = (int)(180 + theta_d / PI*180.0);
				if (theta_z == 360) theta_z = 0;
				if(theta_z >=0 && theta_z<=180)
					theta_z = 180 - theta_z;
				else if(theta_z > 180 && theta_z < 360)
					theta_z = 540 - theta_z;

				double dis = sqrt((i - w/2)*(i - w/2) + (j - h/2)*(j - h/2));
				if (dis < Angle[theta_z])
				{
					Angle[theta_z] = dis;
				}
			}
		}
	}

	/*3. smooth the data*/
	double Angle_f[360];
	linearSmooth(Angle,Angle_f,360);

	Angle[0]=(Angle_f[359]+Angle_f[0]+Angle_f[1])/3;
	Angle[359]=(Angle_f[358]+Angle_f[359]+Angle_f[0])/3;
	for(int i=1;i<359;i++)
	    Angle[i]=(Angle_f[i-1]+Angle_f[i]+Angle_f[i+1])/3;	

	/*4. find hollow*/
	/*for(int i=0;i<360;i++)
	{
		TravDir[i] = 1 - Angle[i]/DMax[i];
       	if(TravDir[i] < 0.0)
    		TravDir[i] = 0.0;
    	if(TravDir[i] > 1.0)
    		TravDir[i] = 1.0;	
	}*/

	/*vector<double> vecAngle(TravDir, TravDir+sizeof(TravDir)/sizeof(double)); 
	vector<int> peakPositions;
	findPeak(vecAngle,peakPositions);

	P item[peakPositions.size()];
	for(vector<int>::size_type i = 0; i !=peakPositions.size(); i++)
	{
	  item[i]=make_pair(peakPositions[i],TravDir[peakPositions[i]]);
	}
	sort(item, item + peakPositions.size(), cmp);
	for (int i = 0; i < peakPositions.size(); i++)
        cout << "(" << item[i].first << ", " << item[i].second << ")\t";*/
    
	/*5. find road contour*/
	Mat roadContour(src.size(),CV_8UC1,Scalar(0));
	
	for(int i=0;i<360;i++)
	{
	    int x = Angle[i]*cos(i*PI/180)+w/2;
    	int y = Angle[i]*sin(i*PI/180)+h/2;
    	x = x<0?0:x;
    	x = x>w-1?w-1:x;
    	y = y<0?0:y;
    	y = y>h-1?h-1:y;

    	int iprev = i==0?359:i-1;
    	int xprev = Angle[iprev]*cos(iprev*PI/180)+w/2;
    	int yprev = Angle[iprev]*sin(iprev*PI/180)+h/2;
    	xprev = xprev<0?0:xprev;
    	xprev = xprev>w-1?w-1:xprev;
    	yprev = yprev<0?0:yprev;
    	yprev = yprev>h-1?h-1:yprev;

    	//roadContour.at<unsigned char>(y,x)=255;
    	line(roadContour,Point(x,y),Point(xprev,yprev),Scalar(255));

    	/* if( i==peakPositions[t] )
    	{
    	 	line(roadContour,Point(x,y),Point(w/2,h/2),Scalar(200));
    	}*/
	}
	imshow("roadContour",roadContour);
	//imwrite("roadContour.png",roadContour);

	/*6. IPM*/
	Mat ipm_img;
	IPMprocess(roadContour, ipm_img);

	/*calculate ipm distance from Point(col/2, row-1)*/	
	double ipmangle[181];
	for(int i=0;i<=181;i++)
		ipmangle[i]=9999.0;

	for(int i=0;i<ipm_img.rows;i++)
	{
		for(int j=0;j<ipm_img.cols;j++)
		{
			if(ipm_img.at<unsigned char>(i,j)==255)
			{
				double ipmdis = sqrt((j-ipm_img.cols/2)*(j-ipm_img.cols/2)
				         	        +(ipm_img.rows-1-i)*(ipm_img.rows-1-i));
				
				int theta_r = (int)( atan2((double)(ipm_img.rows-1-i), (double)(j-ipm_img.cols/2)) * 180 / PI );

				if(ipmdis<ipmangle[theta_r])
					ipmangle[theta_r]=ipmdis;
			}
		}
	}

	for(int i=0;i<ipm_img.rows;i++)
	{
		for(int j=0;j<ipm_img.cols;j++)
		{
			double ipmdis = sqrt((j-ipm_img.cols/2)*(j-ipm_img.cols/2)
			         	        +(ipm_img.rows-1-i)*(ipm_img.rows-1-i));
			
			int theta_r = (int)( atan2((double)(ipm_img.rows-1-i), (double)(j-ipm_img.cols/2)) * 180 / PI );

			if(ipmangle[theta_r]==9999)
			{
				for(int k=theta_r;k>=0;k--)
				{
					if(ipmangle[k]<9999)
					{
						theta_r=k;
						break;
					}
				}
			}

			if(ipmdis>ipmangle[theta_r])
				ipm_img.at<unsigned char>(i,j)=0;
			else
				ipm_img.at<unsigned char>(i,j)=255;
			
		}
	}
	imshow("ipm_img",ipm_img);
	//imwrite("ipm_img.png",ipm_img);

	/*draw ipmgraph*/
	Mat graph1(250, 400, CV_8UC1, Scalar(80));
	line(graph1, Point(10,230),Point(380,230),Scalar(255));
	line(graph1, Point(10,230),Point( 10, 10),Scalar(255));

	int x1,x2,y1,y2,prev=0;
	for(int i=0;i<181;i++)
	{
		if(ipmangle[i]==9999)
			continue;
		
		x1=i+20;
		y1=210 - ipmangle[i];
		circle(graph1, Point(x1,y1), 1, Scalar(255));
		line(graph1,Point(x1,y1),Point(prev+20,210 - ipmangle[prev]),Scalar(255));
		prev=i;		
	}
	imshow("ipm graph",graph1);

	/*find intersection center, saved in 'Point maxLoc()'*/
	float maxValue=0;
	Point maxLoc(0,0); 
	Mat imageThin(ipm_img.size(),CV_32FC1);
    distanceTransform(ipm_img,imageThin,CV_DIST_L2,3);
    Mat distShow;  
    distShow=Mat::zeros(ipm_img.size(),CV_8UC1); 
    for(int i=0;i<imageThin.rows;i++)  
    {  
        for(int j=0;j<imageThin.cols;j++)  
        {  
            distShow.at<uchar>(i,j)=imageThin.at<float>(i,j);  
            if(imageThin.at<float>(i,j)>maxValue)  
            {  
                maxValue=imageThin.at<float>(i,j); 
                maxLoc=Point(j,i);  //坐标  
            }  
        }  
    }  
    normalize(distShow,distShow,0,255,CV_MINMAX);
    ipm_img.at<unsigned char>(maxLoc.y,	maxLoc.x)=0; 
    distShow.at<unsigned char>(maxLoc.y,maxLoc.x)=0;  
    imshow("distanceTransform & maxLoc",distShow);
    //imwrite("distanceTransform.bmp",distShow);
    cout<<"intersection is at "<<(distShow.rows-1-maxLoc.y)*0.04<<" meters ahead of the vehicle."<<endl;

    findTopo(distShow, maxLoc);

	/*draw ipm distances from maxLoc point*/
	// double ipm_graph[360];
	// for(int i=0;i<360;i++)
	// 	ipm_graph[i]=0;

	// for (int i = 0; i < ipm_img.cols; i++)
	// {
	// 	for (int j = 0; j < ipm_img.rows; j++)
	// 	{
	// 		if (ipm_img.at<unsigned char>(j, i)==255)
	// 		{
	// 			int beta = (int)(180 + atan2(maxLoc.y - j, i - maxLoc.x) / PI*180.0);

	// 			if (beta == 360) beta = 0;
	// 			if(beta >=0 && beta<=180)
	// 				beta = 180 - beta;
	// 			else if(beta > 180 && beta < 360)
	// 				beta = 540 - beta;
 
	// 			double dis = sqrt((i - maxLoc.x)*(i - maxLoc.x) + (j - maxLoc.y)*(j - maxLoc.y));
	// 			if (dis > ipm_graph[beta])
	// 			{
	// 				ipm_graph[beta] = dis;
	// 			}
	// 		}
	// 	}
	// }
	// double ipm_graph_f[360];
	// linearSmooth(ipm_graph,ipm_graph_f,360);
	// drawGraph(360, ipm_graph_f);
	
	waitKey(0);
}

void TD_detector::calMaxDistance()
{
	DMax[0]  = DMax[180] = w/2;
	DMax[90] = DMax[270] = h/2;

	double theta = atan2(h/2, w/2);
	int theta_z = (int)(theta / PI*180.0);

	for(int i=0;i<360;i++)
	{
		if( i>0 && i<90 )
		{
			if( i<theta_z )
				DMax[i] = w/2 / cos(i*PI/180);
			else
				DMax[i] = h/2 / sin(i*PI/180);
		}
		if( i>90 && i<180 )
		{
			if( i<180-theta_z )
				DMax[i] = h/2 / cos((i-90)*PI/180);
			else
				DMax[i] = w/2 / sin((i-90)*PI/180);
		}
		if( i>180 && i<270 )
		{
			if( i<180+theta_z )
				DMax[i] = w/2 / cos((i-180)*PI/180);
			else
				DMax[i] = h/2 / sin((i-180)*PI/180);
		}
		if( i>270 && i<360 )
		{
			if( i<360-theta_z )
				DMax[i] = h/2 / cos((i-270)*PI/180);
			else
				DMax[i] = w/2 / sin((i-270)*PI/180);
		}
	}
}

void TD_detector::drawgraph()
{
	Mat graph(250, 400, CV_8UC1, Scalar(80));
	line(graph, Point(10,230),Point(380,230),Scalar(255));
	line(graph, Point(10,230),Point( 10, 10),Scalar(255));

	int x1,x2,y1,y2;
	for(int i=1;i<360;i++)
	{
		x1=(i-1)+20;
		x2=i+20;
		y1=210 - TravDir[i-1]*200;
		y2=210 - TravDir[i]*200;
		line(graph, Point(x1,y1),Point(x2,y2),Scalar(255));
	}

	/*find peak and draw results*/
	// vector<double> vecAngle(TravDir, TravDir+sizeof(TravDir)/sizeof(double)); 
	// vector<int> peakPositions;
	// findPeak(vecAngle,peakPositions);

	// cout<<"Peak positions are: "<<peakPositions.size()<<endl;
	// for(vector<int>::size_type i = 0; i !=peakPositions.size(); i++)
	// {
	// 	cout<<"No."<<i<<": "<<peakPositions[i]<<" "<<1-TravDir[peakPositions[i]]<<endl;
	// 	line(graph, Point(20+peakPositions[i],230),Point(20+peakPositions[i],210-TravDir[peakPositions[i]]*200),Scalar(200));
	// } 

	resize(graph,graph,Size(1.5*400,1.5*250));
	imshow("TD results",graph);
	waitKey(0);
}

void TD_detector::drawGraph(int x, double y[])
{
	Mat graph(250, 400, CV_8UC1, Scalar(80));
	line(graph, Point(10,230),Point(380,230),Scalar(255));
	line(graph, Point(10,230),Point( 10, 10),Scalar(255));

	int prex,prey,curx,cury;
	for(int i=1; i<x; i++)
	{
		prex = i - 1 + 20;
		prey = 210 - y[i-1] ;//* 200;

		curx = i + 20;		
		cury = 210 - y[i] ;//* 200;

		line(graph, Point(curx,cury), Point(prex,prey), Scalar(255));
	}

	resize(graph,graph,Size(1.5*400,1.5*250));
	imshow("TD results",graph);
	waitKey(0);
}
