#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>

#include "Generic.hpp"

namespace tt = thymio_tracker;


class VideoSource
{
public:
	//constructor with calibration
    VideoSource();
	
	//acquire a new frame
	virtual void grabNewFrame()=0;
    void resizeImage();
	
	//get pointer to the frame previously acquired (no copy)
    const cv::Mat& getFramePointer() const {return resized ? imgResized : img;};

    //if want image to be resized, cause the image to be resized and intrinsic calibration to be changed accordingly
    void resizeSource(cv::Size _newSize);
    //same but using ratio
    void resizeSource(float _r);
    bool isOver(){return false;}

 //attributes:
    //current image
    cv::Mat img;
    
    //resized img if resized
    cv::Mat imgResized;
    
    //did we ask the image to be resized
    bool resized;
    cv::Size img_size;
};

class VideoSourceSeq : public VideoSource
{
public:
    //constructor with calibration: parameters are the sequence filename in sprintf style, id0 is the first frame number
    VideoSourceSeq(const char *_printfPath, int id0=0);
    
    void grabNewFrame();
    int getFrameId(){return frameId;};
    bool isOver(){return end_sequence;}
    
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
    VideoSourceLive();
    
    void grabNewFrame();
    
private:
    
    cv::VideoCapture captureDevice;
};

