
#include "ThymioTracker.h"

#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;

int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/calibration/embedded_camera_calib.xml",
        "../data/GHscale_Arth_Perspective.dat",
        {
            "../data/marker.xml.gz",
            // "../data/marker2.xml.gz",
            // "../data/ziggu.xml.gz"
        }
        );
    
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
        
        std::cout << "\33[2K\r" << tracker.getTimer().getFps() << "fps" << std::flush;
        
        // const tt::DetectionInfo& di = tracker.getDetectionInfo();
        // const tt::Landmark& landmark = tracker.getLandmarks()[0];
        // cv::Mat h = di.landmarkDetections[0].getHomography();
        // if(!h.empty())
        // {
        //     cv::Mat crop;
        //     cv::Size size = landmark.getImage().size();
        //     cv::warpPerspective(inputImage, crop, h.inv(), size);
        //     imshow(window_name, crop);
        // }
        
        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}
