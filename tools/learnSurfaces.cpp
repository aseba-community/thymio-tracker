/* In this program, an image sequence which the robot placed on an Aruco board is used as well as 
the robot GT position computed using ARuco lib, to learn the appearance of the robot. 
Several 3D planar surfaces have been defined in ThymioBlobModel from Models.hpp, in this program 
we learn their texture by backprojecting the images from the sequence into the surface and averaging them*/

#include "ThymioTracker.h"
#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;


//work offline on recorded sequence
int main(int argc, char** argv)
{
    int firstFrame = 2;
    tt::ThymioBlobModel mRobot;
    std::cout<<"allocate"<<std::endl;
    mRobot.allocateSurfaceLearning();

    //get sequence
    VideoSourceSeq videoSource("/Users/amaurydame/Data/Thymio/onBoard/seq/image-%04d.png",NexusCam,firstFrame+2);
    videoSource.resizeSource(0.5);

    //get pose computed in aruco
    std::vector<int> founds;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;

    cv::FileStorage store("/Users/amaurydame/Data/Thymio/onBoard/thymioOnBoard.bin", cv::FileStorage::READ);
    //cv::FileStorage store("/Users/amaurydame/Data/Thymio/onBoard/thymioOnBoard_offCentered.bin", cv::FileStorage::READ);
    //cv::FileStorage store("/Users/amaurydame/Libs/aruco-1.3.0/build/utils/detections.bin", cv::FileStorage::READ);
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

    
    while(!videoSource.isOver())
    //while(cpt<10)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        //use board pose to compute robots pose
        cv::Affine3d boardPose = cv::Affine3d(rvecs[cpt],tvecs[cpt]);
        cv::Affine3d robotPose = boardPose*cv::Affine3d().translate(cv::Vec3d(0.,0.,-0.022))*cv::Affine3d().rotate(cv::Vec3d(0.,M_PI,0.));

        //tracker.update(inputGray);
        if(founds[cpt]==1)
        {
            mRobot.learnAppearance(inputGray, videoSource.mCalibration, robotPose);
            mRobot.draw(inputImage, videoSource.mCalibration, robotPose);
        }
            
        
        imshow(window_name, inputImage);
        cpt ++;
        

        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q')
            break;
    }
    mRobot.writeSurfaceLearned();

    
    return 0;
}
