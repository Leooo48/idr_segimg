#pragma once
#include "unary.h"
#include "labelcompatibility.h"
#include "objective.h"
#include "pairwise.h"
#include <vector>

/*by leo*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
using namespace cv;

/**** DenseCRF ****/
class DenseCRF{
protected:
	// Number of variables and labels
	int N_, M_; 
	
	// Store the unary term
	UnaryEnergy * unary_;
	
	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;
	
	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}
public:
	// Create a dense CRF model of size N with M labels
	DenseCRF( int N, int M );
	virtual ~DenseCRF();
	
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	// (ownership of LabelCompatibility will be transfered to this class)
	void addPairwiseEnergy( const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );
	
	// Add your own favorite pairwise potential (ownership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential );
	
	// Set the unary potential (ownership will be transfered to this class)
	void setUnaryEnergy( UnaryEnergy * unary );
	// Add a constant unary term
	void setUnaryEnergy( const MatrixXf & unary );
	// Add a logistic unary term
	void setUnaryEnergy( const MatrixXf & L, const MatrixXf & f );
	
	// Run inference and return the probabilities
	MatrixXf inference( int n_iterations ) const;
	
	// Run MAP inference and return the map for each pixel
	VectorXs map( int n_iterations ) const;
	
	// Step by step inference
	MatrixXf startInference() const;
	void stepInference( MatrixXf & Q, MatrixXf & tmp1, MatrixXf & tmp2 ) const;
	VectorXs currentMap( const MatrixXf & Q ) const;
	
	// Learning functions
	// Compute the gradient of the objective function over mean-field marginals with
	// respect to the model parameters
	double gradient( int n_iterations, const ObjectiveFunction & objective, VectorXf * unary_grad, VectorXf * lbl_cmp_grad, VectorXf * kernel_grad=NULL ) const;
public: /* Debugging functions */
	// Compute the unary energy of an assignment l
	VectorXf unaryEnergy( const VectorXs & l );
	
	// Compute the pairwise energy of an assignment l (half of each pairwise potential is added to each of it's endpoints)
	VectorXf pairwiseEnergy( const VectorXs & l, int term=-1 );
	
	// Compute the KL-divergence of a set of marginals
	double klDivergence( const MatrixXf & Q ) const;

public: /* Parameters */
	VectorXf unaryParameters() const;
	void setUnaryParameters( const VectorXf & v );
	VectorXf labelCompatibilityParameters() const;
	void setLabelCompatibilityParameters( const VectorXf & v );
	VectorXf kernelParameters() const;
	void setKernelParameters( const VectorXf & v );
};

class DenseCRF2D:public DenseCRF{
protected:
	// Width, height of the 2d grid
	int W_, H_;
public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF2D( int W, int H, int M );
	virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian( float sx, float sy, LabelCompatibility * function=NULL, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );
	
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, LabelCompatibility * function=NULL, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC );
	
	// Set the unary potential for a specific variable
	using DenseCRF::setUnaryEnergy;

        //some other tools for application by leo
        int nColors;
        int colors[255];
        unsigned char* convert2ppm(cv::Mat img_png);
        cv::Mat ppm2mat(unsigned char * res, int width, int height);
        VectorXs getLabeling( const unsigned char * im, int N, int M, int colors[], int nColors);
        unsigned char * colorize( const VectorXs & labeling, int W, int H, int colors[]);
};
