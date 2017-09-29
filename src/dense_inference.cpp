#include "densecrf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>

using namespace cv;

// // Store the colors we read, so that we can write them again.
// int nColors = 0;
// int colors[255];
// int getColor( const unsigned char * c ){
// 	return c[0] + 256*c[1] + 256*256*c[2];
// }

// void putColor( unsigned char * c, int cc ){
// 	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
// }
// // Produce a color image from a bunch of labels
// unsigned char * colorize( const VectorXs & labeling, int W, int H, int& colors ){
// 	unsigned char * r = new unsigned char[ W*H*3 ];
// 	for( int k=0; k<W*H; k++ ){
// 		int c = colors[ labeling[k] ];
// 		putColor( r+3*k, c );
// 	}
// 	//printf("%d %d %d \n",r[0],r[1],r[2]);
// 	return r;
// }

// // Read the labeling from a file
// VectorXs getLabeling( const unsigned char * im, int N, int M, int& colors, int nColors ){
// 	VectorXs res(N);
// 	//printf("%d %d %d \n",im[0],im[1],im[2]);
// 	for( int k=0; k<N; k++ ){
// 		// Map the color to a label
// 		int c = getColor( im + 3*k );
// 		int i;
// 		for( i=0;i<nColors && c!=colors[i]; i++ );
// 		if (c && i==nColors){
// 			if (i<M)
// 				colors[nColors++] = c;
// 			else
// 				c=0;
// 		}
// 		res[k] = c?i:-1;
// 	}
// 	return res;
// } 

// unsigned char* convert2ppm(Mat& img_png)
// {
// 	int width  = img_png.cols;
// 	int height = img_png.rows;
// 	unsigned char * r = new unsigned char[width*height*3];

// 	int count = 0;
// 	for(int i=0; i < height; i++)
// 	{
// 		for(int j=0; j < width; j++)
// 		{
// 			if(count < width*height*3)
// 				r[count++] = img_png.at<Vec3b>(i,j)[0];
// 			if(count < width*height*3)
// 				r[count++] = img_png.at<Vec3b>(i,j)[1];
// 			if(count < width*height*3)
// 				r[count++] = img_png.at<Vec3b>(i,j)[2];
// 		}
// 	}

// 	return r;
// }

// Mat ppm2mat(unsigned char * res, int width, int height)
// {
// 	Mat img_png(height,width,CV_8UC3,Scalar(0,0,0));

// 	int count = 0;
// 	for(int i=0; i < height; i++)
// 	{
// 		for(int j=0; j < width; j++)
// 		{
// 			if(count < width*height*3)
// 				img_png.at<Vec3b>(i,j)[0] = res[count++];
// 			if(count < width*height*3)
// 				img_png.at<Vec3b>(i,j)[1] = res[count++];
// 			if(count < width*height*3)
// 				img_png.at<Vec3b>(i,j)[2] = res[count++];
// 		}
// 	}

// 	return img_png;
// }

// Certainty that the groundtruth is correct
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

int main(){            
	
	// Number of labels
	const int M = 12;
	
	// Load the color image and some crude annotations
	int W, H, GW, GH;

	Mat im_png = imread("/home/leo/cross_leo/data/000027.png");
	W = im_png.cols;
	H = im_png.rows;
	Mat anno_png = imread("/home/leo/cross_leo/data/segnet.png");
	GW = anno_png.cols;
	GH = anno_png.rows;

	if (W!=GW || H!=GH){
		printf("Annotation size doesn't match image!\n");
		return 1;
	}

	// Setup the CRF model
	DenseCRF2D crf(W, H, M);

    unsigned char * im;
    unsigned char * anno;
	im = crf.convert2ppm(im_png);
	anno = crf.convert2ppm(anno_png);
	
	crf.nColors=0;
	/////////// Put your own unary classifier here! ///////////
	MatrixXf unary = computeUnary( crf.getLabeling( anno, W*H, M, crf.colors, crf.nColors), M );
	///////////////////////////////////////////////////////////

	// Specify the unary potential as an array of size W*H*(#classes)
	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
	crf.setUnaryEnergy( unary );
	// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
	// x_stddev = 3
	// y_stddev = 3
	// weight = 3
	crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 3 ) );
	// add a color dependent term (feature = xyrgb)
	// x_stddev = 60
	// y_stddev = 60
	// r_stddev = g_stddev = b_stddev = 20
	// weight = 10
	crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new PottsCompatibility( 10 ) );
	
	// Do map inference
// 	MatrixXf Q = crf.startInference(), t1, t2;
// 	printf("kl = %f\n", crf.klDivergence(Q) );
// 	for( int it=0; it<5; it++ ) {
// 		crf.stepInference( Q, t1, t2 );
// 		printf("kl = %f\n", crf.klDivergence(Q) );
// 	}
// 	VectorXs map = crf.currentMap(Q);
	VectorXs map = crf.map(5);
	// Store the result
	unsigned char *res = crf.colorize( map, W, H, crf.colors);

	Mat res_png(H,W,CV_8UC3,Scalar(0,0,0));
	res_png = crf.ppm2mat(res, W, H);
	imwrite("/home/leo/cross_leo/data/res.png",res_png);

	delete[] im;
	delete[] anno;
	delete[] res;
}
