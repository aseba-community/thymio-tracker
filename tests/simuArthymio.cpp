/*  program to test the landmark and robot tracker online

the progrom search in ../data for Config.xml file where
the file path of the camera calibration, geometric hashing file, robot surfaces
and landmarks files are defined. */

#include "ThymioTracker.h"
#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;


//work offline on recorded sequence
int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/");

    VideoSourceLive videoSource;
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );

    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();


        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        tracker.update(inputGray);
        tracker.drawLastDetection(&inputImage);
        
        imshow(window_name, inputImage);
        
        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q' || videoSource.isOver())
            break;
    }
    
    return 0;
}
