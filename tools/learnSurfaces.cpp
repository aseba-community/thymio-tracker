/* 

In this program, an image sequence which the robot placed on an Aruco board is used as well as 
the robot GT position computed using ARuco lib, to learn the appearance of the robot. 
Several 3D planar surfaces have been defined in ThymioBlobModel from Models.hpp, in this program 
we learn their texture by backprojecting the images from the sequence into the surface and averaging them

Default usage:
learnSurfaces ../data/calibration/nexus_camera_calib.xml /Users/amaurydame/Data/Thymio/onBoard/seq/image-%04d.png /Users/amaurydame/Data/Thymio/onBoard/thymioOnBoard.bin modelSurfaces.xml 2

*/

#include "ThymioTracker.h"
#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;

void print_usage(const char* command)
{
    std::cerr << "Usage:\n\t" << command << " <calib file> <seq files> <pose file> <output file> [<offset>]" << std::endl;
}

//work offline on recorded sequence
int main(int argc, char** argv)
{
    if(argc < 5 || argc > 6)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string calibFilename = argv[1];
    std::string seqFilename = argv[2];
    std::string poseFilename = argv[3];
    std::string outFilename = argv[4];
    
    int offset = 0;
    if(argc >= 6)
        offset = std::stoi(argv[5]);


    int firstFrame = 2;
    tt::ThymioBlobModel mRobot;
    //allocate space to the surfaces to store the texture
    mRobot.allocateSurfaceLearning();

    //get sequence
    VideoSourceSeq videoSource(seqFilename.c_str(),firstFrame+offset);
    videoSource.resizeSource(0.5);

    //get calib
    tt::IntrinsicCalibration mCalibration(calibFilename);

    //for each image of the sequence, we have:
    std::vector<int> founds;//pose computed in the image or not (sometimes the board is not found)
    std::vector<cv::Vec3d> rvecs;//pose of the board (not the robot !)
    std::vector<cv::Vec3d> tvecs;

    //read the xml file which stores this values
    cv::FileStorage store(poseFilename, cv::FileStorage::READ);
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
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        //use board pose to compute robots pose
        //here the robot was placed in the middle of the board
        cv::Affine3d boardPose = cv::Affine3d(rvecs[cpt],tvecs[cpt]);
        cv::Affine3d robotPose = boardPose*cv::Affine3d().translate(cv::Vec3d(0.,0.,-0.022))*cv::Affine3d().rotate(cv::Vec3d(0.,M_PI,0.));

        if(founds[cpt]==1)
        {
            mRobot.learnAppearance(inputGray, mCalibration, robotPose);
            mRobot.draw(inputImage, mCalibration, robotPose);
        }
            
        
        imshow(window_name, inputImage);
        cpt ++;
        

        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q')
            break;
    }

    //save the model to an xml file
    mRobot.writeSurfaceLearned(outFilename);

    
    return 0;
}
