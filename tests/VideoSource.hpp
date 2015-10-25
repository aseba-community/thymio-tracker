#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>

#include "Generic.hpp"

using namespace std;
using namespace cv;

enum CameraType {NexusCam,
		  EmbeddedCam,
		  UndefinedCam
};

class videoSource
{
 public:
	//constructor with calibration
	videoSource(CameraType _camType=UndefinedCam);
	
	//acquire a new frame
	virtual void grabNewFrame()=0;
    void resizeImage();
	
	//get pointer to the frame previously acquired (no copy)
    cv::Mat* GetFramePointer(){return resized?&imgResized:&img;};

    //if want image to be resized, cause the image to be resized and intrinsic calibration to be changed accordingly
    void resizeSource(Size _newSize);
    //same but using ratio
    void resizeSource(float _r);

 //attributes:
    //camera intrinsic calibration
    IntrinsicCalibration mCalibration;
    
    //current image
    Mat img;
    //resized img if resized
    Mat imgResized;
    //output img size
    Size imgSize;
    
    //did we ask the image to be resized
    bool resized;
};

class videoSourceSeq : public videoSource
{
public:
    //constructor with calibration
    videoSourceSeq(char *_printfPath,CameraType _camType,int id0=0);
    
    void grabNewFrame();
private:
    //where to read image sequence
    char *printfPath;
    //current index read
    int frameId;
    //have we reached the end
    bool end_sequence;
};

class videoSourceLive : public videoSource
{
public:
    //constructor with calibration
    videoSourceLive(CameraType _camType);
    
    void grabNewFrame();
private:
    VideoCapture captureDevice;
};

