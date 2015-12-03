
#include "arthymio.h"

#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;

int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/calibration/embedded_camera_calib.xml",
                              "../data/GHscale_Arth_Perspective.dat");
    
    videoSourceLive mVideoSource(EmbeddedCam);
    mVideoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        mVideoSource.grabNewFrame();
        cv::Mat& inputImage = *mVideoSource.GetFramePointer();
        
        tracker.update(inputImage);
        tracker.drawLastDetection(&outputImage);
        
        imshow(window_name, outputImage);
        
        if(waitKey(5) == 27)
            break;
    }
    
    return 0;
}
