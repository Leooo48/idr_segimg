// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <vector>
#include <iostream>  
#include <string>  
#include <list>   
#include <map>  
#include <stack> 

class TD_detector
{
private:
    int w, h;
    double Angle[360]={0};
    double DMax[360]={0};

public:
    TD_detector(int img_width, int img_height)
    {
        w = img_width;
        h = img_height;
    }

    double TravDir[360]={0};

    void calMaxDistance();

    void getNGpoints(cv::Mat src);
    void drawgraph();
    void drawGraph(int x, double y[]);
};
