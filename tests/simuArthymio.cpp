
#include "ThymioTracker.h"
#include "VideoSource.hpp"

static const char window_name[] = "Tracker";

namespace tt = thymio_tracker;


//work offline on recorded sequence
/*int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/");

    //VideoSourceLive videoSource(EmbeddedCam);
    VideoSourceSeq videoSource("/Users/amaurydame/Data/Thymio/landmark/Landmark2/image-%03d.png",EmbeddedCam,1);
    //VideoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    //videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        tracker.update(inputGray);
        tracker.drawLastDetection(&inputImage);
        
        imshow(window_name, inputImage);
        
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
*/

//work online with embedded camera
int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/");
    
    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        tracker.update(inputGray);
        tracker.drawLastDetection(&inputImage);
        
        imshow(window_name, inputImage);
        std::cout<<inputGray.size().width<<"x"<<inputGray.size().height<<std::endl;
        
        std::cout << "\33[2K\r" << tracker.getTimer().getFps() << "fps" << std::flush;

        auto key = cv::waitKey(5);
        //auto key = cv::waitKey(100);
        //auto key = cv::waitKey();
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}


//offline with orientation
/*int main(int argc, char** argv)
{
    tt::ThymioTracker tracker("../data/");

    VideoSourceSeq videoSource("/Users/amaurydame/Data/Thymio/orientation/rec1/img_%d.png",NexusCam,0);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();

        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        //load corresponding orientation
        cv::Mat deviceOrientation;
        std::string path = "/Users/amaurydame/Data/Thymio/orientation/rec1/";
        std::string name2 = path+"ori_" + std::to_string(videoSource.getFrameId()) + ".dat";
        cv::FileStorage fs(name2, cv::FileStorage::READ);
        cv::read(fs["deviceOrientation"], deviceOrientation);

        tracker.update(inputGray);
        tracker.drawLastDetection(&inputImage,&deviceOrientation);
        
        imshow(window_name, inputImage);
        
        //auto key = cv::waitKey(5);
        auto key = cv::waitKey();
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}*/
