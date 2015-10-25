//Copyright 2008 Isis Innovation Limited 
#include "VideoSource.hpp"
#include "Generic.hpp"

#include <unistd.h>

videoSource::videoSource(CameraType _camType)
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
    
    imgSize.width = (int) fs["image_width"];
    imgSize.height = (int) fs["image_height"];
    fs["camera_matrix"] >> mCalibration.cameraMatrix;
    fs["distortion_coefficients"] >> mCalibration.distCoeffs;
}

void videoSource::resizeSource(Size _newSize)
{
    resized=true;
    
    //have to change camera calibration
    resizeCameraMatrix(mCalibration.cameraMatrix,imgSize,_newSize);
    
    //change output image size
    imgSize=_newSize;
}
void videoSource::resizeSource(float _r)
{
    resized=true;
    
    //change output iamge size
    Size sourceSize=imgSize;
    imgSize.width=_r*sourceSize.width;imgSize.height=_r*sourceSize.height;

    //have to change camera calibration
    resizeCameraMatrix(mCalibration.cameraMatrix,sourceSize,imgSize);
}

void videoSource::resizeImage()
{
    resize(img, imgResized, imgSize, 0, 0);
}

videoSourceLive::videoSourceLive(CameraType _camType):videoSource(_camType)
{
    captureDevice.open(0);
}

void videoSourceLive::grabNewFrame()
{
    captureDevice>>img;
    if(resized)resizeImage();
}


videoSourceSeq::videoSourceSeq(char *_printfPath,CameraType _camType,int id0):videoSource(_camType)
{
    frameId=id0;
    printfPath=_printfPath;
    end_sequence=false;
}


void videoSourceSeq::grabNewFrame()
{
    if(!end_sequence)
    {
        frameId++;
        char fileName[200];
        sprintf(fileName, printfPath, frameId);
        
        std::ifstream fout;
        fout.open(fileName);
        
        if(!fout.is_open())
        {
            std::cerr<<"VideoSourceSeq : End sequence"<<std::endl;
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
