//Copyright 2008 Isis Innovation Limited 
#include "VideoSource.hpp"
#include "Generic.hpp"

#include <unistd.h>

using namespace std;
using namespace cv;

VideoSource::VideoSource(CameraType _camType)
{
    resized=false;
    
    char cwd[200];
    getcwd(cwd, 199);
    std::cerr << cwd << std::endl;
    
    FileStorage fs;
    switch (_camType)
    {
        case NexusCam:
            fs.open("../data/calibration/nexus_camera_calib.xml", FileStorage::READ);
            break;
        case EmbeddedCam:
            fs.open("../data/calibration/embedded_camera_calib.xml", FileStorage::READ);
            break;
        default:
            fs.open("../data/calibration/default_camera_calib.xml", FileStorage::READ);
            break;
    }
    if(!fs.isOpened())
        throw std::runtime_error("Calibration file not found!");
    
    mCalibration.imageSize.width = (int) fs["image_width"];
    mCalibration.imageSize.height = (int) fs["image_height"];
    fs["camera_matrix"] >> mCalibration.cameraMatrix;
    fs["distortion_coefficients"] >> mCalibration.distCoeffs;
    
    fs.release();
}

void VideoSource::resizeSource(Size _newSize)
{
    resized=true;
    
    //have to change camera calibration
    tt::resizeCameraMatrix(mCalibration.cameraMatrix, mCalibration.imageSize, _newSize);
    
    //change output image size
    mCalibration.imageSize=_newSize;
}
void VideoSource::resizeSource(float _r)
{
    resized=true;
    
    //change output iamge size
    Size sourceSize=mCalibration.imageSize;
    mCalibration.imageSize.width=_r*sourceSize.width;
    mCalibration.imageSize.height=_r*sourceSize.height;

    //have to change camera calibration
    tt::resizeCameraMatrix(mCalibration.cameraMatrix,sourceSize,mCalibration.imageSize);
}

void VideoSource::resizeImage()
{
    resize(img, imgResized, mCalibration.imageSize, 0, 0);
}

VideoSourceLive::VideoSourceLive(CameraType _camType):VideoSource(_camType)
{
    captureDevice.open(0);
}

void VideoSourceLive::grabNewFrame()
{
    captureDevice>>img;
    if(resized)resizeImage();
}

VideoSourceSeq::VideoSourceSeq(const char *_printfPath, CameraType _camType, int id0)
    : VideoSource(_camType)
    , printfPath(_printfPath)
{
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

    if(img.size().width != mCalibration.imageSize.width)
        rescaleCalibration(mCalibration,img.size());

    frameId=id0;
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
