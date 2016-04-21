
#include "ThymioTracker.h"

#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;


//work online with embedded camera
int main(int argc, char** argv)
{

    tt::ThymioTracker tracker("../data/calibration/default_camera_calib.xml",
        "../data/",
        {
            //"../data/landmarks/markerTest.xml.gz",
            "../data/landmarks/marker.xml.gz",
            "../data/landmarks/ziggu.xml.gz",
            // "../data/marker2.xml.gz",
            // "../data/ziggu.xml.gz"
        }
        );
    
    VideoSourceLive videoSource(EmbeddedCam);
    //VideoSourceSeq videoSource("/Users/amaurydame/Data/Thymio/landmark/Landmark2/image-%03d.png",EmbeddedCam,1);
    //VideoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    bool calibrate = false;
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        tracker.updateLandmarks(inputGray);
        tracker.updateCalibration();
        tracker.drawLastDetection(&inputImage);
        
        imshow(window_name, inputImage);

        const tt::CalibrationInfo& ci = tracker.getCalibrationInfo();
        std::cout<<ci.getNbFramesToCalibration()<<std::endl;
        if(!calibrate && ci.getNbFramesToCalibration() == 0)
        {
            cv::FileStorage fs("../data/calibration/online_camera_calib.xml", cv::FileStorage::WRITE);
            std::cout<<"store calib"<<std::endl;
            tracker.writeCalibration(fs);
            std::cout<<"store calib passed"<<std::endl;
            calibrate = true;
        }
        
        //std::cout << "\33[2K\r" << tracker.getTimer().getFps() << "fps" << std::flush;
        
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
        //auto key = cv::waitKey();
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}
