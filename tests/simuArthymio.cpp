
#include "ThymioTracker.h"

#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;

int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/calibration/embedded_camera_calib.xml",
                              "../data/GHscale_Arth_Perspective.dat");
    
    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();
        
        tracker.update(inputImage);
        tracker.drawLastDetection(&inputImage);
        
        imshow(window_name, inputImage);
        
        if(cv::waitKey(5) == 27)
            break;
    }
    
    return 0;
}
