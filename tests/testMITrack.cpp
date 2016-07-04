
/* This program is used to debug the robot tracking algorithm which uses the 
planar surface appearances learned in OnBoard.cpp*/

#include "ThymioTracker.h"
#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;


//work offline on recorded sequence
int main(int argc, char** argv)
{
    //test
    cv::Mat mImage;
    cv::Mat *mImagePtr = &mImage;

    mImage.create(3,3, CV_32FC1);
    mImage.setTo(0.);
    std::cout<<*mImagePtr<<std::endl;


    mImage.setTo(1.);
    std::cout<<*mImagePtr<<std::endl;




    int firstFrame = 10;
    tt::ThymioBlobModel mRobot;

    //get sequence
    VideoSourceSeq videoSource("/Users/amaurydame/Data/Thymio/onBoard/seq/image-%04d.png",NexusCam,firstFrame+2);
    videoSource.resizeSource(0.5);

    //get pose computed in aruco
    std::vector<int> founds;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;

    cv::FileStorage store("/Users/amaurydame/Data/Thymio/onBoard/thymioOnBoard.bin", cv::FileStorage::READ);
    cv::FileNode n1 = store["founds"];
    cv::read(n1,founds);
    cv::FileNode n2 = store["rvecs"];
    cv::read(n2,rvecs);
    cv::FileNode n3 = store["tvecs"];
    cv::read(n3,tvecs);
    store.release();

    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    int cpt = firstFrame;

    cv::FileStorage fs("../data/modelSurfaces.xml.gz", cv::FileStorage::READ);
    std::cout<<"readSurfaceLearned"<<std::endl;
    mRobot.readSurfaceLearned(fs);
    bool isRobotposeInit = false;
    
    cv::Affine3d robotPose;
    cv::Mat prevImg;
    while(!videoSource.isOver())
    //while(cpt<10)
    {
        std::cout<<"Frame "<<cpt <<std::endl;
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        //use board pose to compute robots pose
        if(founds[cpt]==1 && !isRobotposeInit)
        {
            cv::Affine3d boardPose = cv::Affine3d(rvecs[cpt],tvecs[cpt]);
            robotPose = boardPose*cv::Affine3d().translate(cv::Vec3d(0.,0.,-0.022))*cv::Affine3d().rotate(cv::Vec3d(0.,M_PI,0.));
            inputGray.copyTo(prevImg);
            isRobotposeInit = true;
        }

        //perform tracking
        if(isRobotposeInit)
        {
            //track
            cv::Affine3d robotPoseTrack;
            mRobot.track(inputGray, prevImg, videoSource.mCalibration, robotPose, robotPoseTrack);

            //display
            //mRobot.draw(inputImage, videoSource.mCalibration, robotPose);
            mRobot.draw(inputImage, videoSource.mCalibration, robotPoseTrack);
            robotPose = robotPoseTrack;
        }
            
        
        imshow(window_name, inputImage);
        inputGray.copyTo(prevImg);
        cpt ++;
        

        auto key = cv::waitKey();
        if(key == 27 || key == 'q')
            break;
    }

    
    return 0;
}
