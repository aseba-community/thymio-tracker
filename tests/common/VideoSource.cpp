//Copyright 2008 Isis Innovation Limited 
#include "VideoSource.hpp"
#include "Generic.hpp"

#include <unistd.h>

using namespace std;
using namespace cv;

VideoSource::VideoSource()
{
    resized=false;
}


void VideoSource::resizeSource(Size _newSize)
{
    resized=true;
    
    //change output image size
    img_size=_newSize;
}

void VideoSource::resizeSource(float _r)
{
    resized=true;
    
    //change output iamge size
    Size sourceSize=img_size;
    img_size.width=_r*sourceSize.width;
    img_size.height=_r*sourceSize.height;
}

void VideoSource::resizeImage()
{
    resize(img, imgResized, img_size, 0, 0);
}

VideoSourceLive::VideoSourceLive()
{
    resized=false;
    captureDevice.open(0);

    //init iamge size
    captureDevice>>img;
    img_size = img.size();
}

void VideoSourceLive::grabNewFrame()
{
    captureDevice>>img;
    if(resized)resizeImage();
}

VideoSourceSeq::VideoSourceSeq(const char *_printfPath, int id0):
    printfPath(_printfPath)
{
    resized=false;

    //if resized version of input has been saved then calibration has to be resized too
    //load first image 
    char fileName[200];
    sprintf(fileName, printfPath.c_str(), id0);
    
    std::ifstream fout;
    fout.open(fileName);
    
    if(!fout.is_open())
        std::cerr<<"VideoSourceSeq : File does not exist"<<std::endl;
    else
        img=imread(fileName);

    frameId=id0;
    img_size = img.size();
    end_sequence=false;
}

void VideoSourceSeq::grabNewFrame()
{
    if(!end_sequence)
    {
        frameId++;
        char fileName[200];
        sprintf(fileName, printfPath.c_str(), frameId);
        
        std::ifstream fout;
        fout.open(fileName);
        
        if(!fout.is_open())
        {
            std::cerr<<"VideoSourceSeq : End sequence"<<std::endl;
            frameId--;

            //reload in case of we printed something over the image
            sprintf(fileName, printfPath.c_str(), frameId);
            img=imread(fileName);
            if(resized)resizeImage();

            end_sequence=true;
        }
        else
        {
            img=imread(fileName);
            if(resized)resizeImage();
        }
        
        fout.close();
    }
}
