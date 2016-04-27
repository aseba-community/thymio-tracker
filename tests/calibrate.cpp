
#include "ThymioTracker.h"

#include "VideoSource.hpp"

static const char window_name[] = "Calibration";

namespace tt = thymio_tracker;

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners)
{
    corners.clear();

    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));


}


//work online with embedded camera
/*int main(int argc, char** argv)
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

        //tracker.update(inputGray);
        //tracker.drawLastDetection(&inputImage);

        std::vector<cv::Point2f> pointBuf;
        cv::Size boardSize(4,11);
        bool found = findCirclesGrid( inputImage, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID );
        drawChessboardCorners( inputImage, boardSize, cv::Mat(pointBuf), found );
        
        imshow(window_name, inputImage);
        
        std::cout << "\33[2K\r" << tracker.getTimer().getFps() << "fps" << std::flush;

        auto key = cv::waitKey(5);
        //auto key = cv::waitKey(100);
        //auto key = cv::waitKey();
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}*/
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
    std::cout<<"Press space to add a frame for calib and Enter to calibrate and save to file..."<<std::endl;

    cv::Size boardSize(4,11);
    std::vector<cv::Point3f> objectCorners;
    calcBoardCornerPositions(boardSize, 0.02, objectCorners);

    //info kept for calib
    std::vector<std::vector<cv::Point3f> > objectPoints;//3d vertices
    std::vector<std::vector<cv::Point2f> > imagePoints;//projection of 3d vertices


    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();
        
        bool found = false;
        std::vector<cv::Point2f> pointBuf;
        
        //process grid
        found = findCirclesGrid( inputImage, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID );
        
        if(found)drawChessboardCorners( inputImage, boardSize, cv::Mat(pointBuf), found );
        
        //plot counter
        char cStr[100];
        sprintf(cStr, "%lu", imagePoints.size());
        putText(inputImage, cStr,
                    cv::Point2i(inputImage.size().width - 40,inputImage.size().height-10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8, cvScalar(250,250,250), 1, CV_AA);

        imshow(window_name, inputImage);
        
        //if(waitKey(30) >= 0) break;
        auto key = cv::waitKey(5);


        if(found && key == 32)//space => keep frame for calib and next frame
        {
            imagePoints.push_back(pointBuf);
            objectPoints.push_back(objectCorners);
        }

        if(key == 13)//enter => calibrate
        {
            std::vector<cv::Mat> rotationVectors;
            std::vector<cv::Mat> translationVectors;

            cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

            int flags = 0;
            double rms = calibrateCamera(objectPoints, imagePoints, inputImage.size(), cameraMatrix,
                          distortionCoefficients, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

            std::cout<<"camera calibration done ... RMS = "<<rms<<std::endl;

            //save to file
            cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
            cv::write(fs,"camera_matrix", cameraMatrix);
            cv::write(fs,"distortion_coefficients", distortionCoefficients);
            cv::write(fs,"image_width", inputImage.size().width);
            cv::write(fs,"image_height", inputImage.size().height);
            fs.release();
            
        }

        if(key == 27 || key == 'q')
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

/*
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

        tracker.updateCalibration(inputGray);
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
}*/
