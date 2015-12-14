#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>

#include "Generic.hpp"

namespace tt = thymio_tracker;

enum CameraType {NexusCam,
		  EmbeddedCam,
		  UndefinedCam
};

class VideoSource
{
public:
	//constructor with calibration
	VideoSource(CameraType _camType=UndefinedCam);
	
	//acquire a new frame
	virtual void grabNewFrame()=0;
    void resizeImage();
	
	//get pointer to the frame previously acquired (no copy)
    const cv::Mat& getFramePointer() const {return resized ? imgResized : img;};

    //if want image to be resized, cause the image to be resized and intrinsic calibration to be changed accordingly
    void resizeSource(cv::Size _newSize);
    //same but using ratio
    void resizeSource(float _r);

 //attributes:
    //camera intrinsic calibration
    tt::IntrinsicCalibration mCalibration;
    
    //current image
    cv::Mat img;
    
    //resized img if resized
    cv::Mat imgResized;
    //output img size
    //Size imgSize; //now in camera intrinsic calibration
    
    //did we ask the image to be resized
    bool resized;
};

class VideoSourceSeq : public VideoSource
{
public:
    //constructor with calibration
    VideoSourceSeq(const char *_printfPath, CameraType _camType, int id0=0);
    
    void grabNewFrame();
    
private:
    //where to read image sequence
    const std::string printfPath;
    //current index read
    int frameId;
    //have we reached the end
    bool end_sequence;
};

class VideoSourceLive : public VideoSource
{
public:
    //constructor with calibration
    VideoSourceLive(CameraType _camType);
    
    void grabNewFrame();
    
private:
    
    cv::VideoCapture captureDevice;
};

