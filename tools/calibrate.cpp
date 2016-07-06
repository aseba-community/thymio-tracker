
//program to calibrate the camera using live images and the asymetric grid from opencv

#include "ThymioTracker.h"
#include "VideoSource.hpp"
#include "opencv2/opencv.hpp"

static const char window_name[] = "Calibration";

namespace tt = thymio_tracker;

using namespace cv;

//create the 3D model of the asymetric grid
static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners)
{
    corners.clear();

    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
}

void print_usage(const char* command)
{
    std::cerr << "Usage:\n\t" << command << " <output calib file>" << std::endl;
}


int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string outputFilename = argv[1];

    std::cout<<"Put the assymetric grid pattern in front of the camera."<<std::endl;
    std::cout<<"When the pattern is detected and plotted, keep it in place until the image is kept for calibration."<<std::endl;
    std::cout<<"When you have enough calibration images, press Enter to calibrate and save to the output xml file..."<<std::endl;

    cv::Size boardSize(4,11);
    std::vector<cv::Point3f> objectCorners;
    calcBoardCornerPositions(boardSize, 0.02, objectCorners);

    //info kept for calib
    std::vector<std::vector<cv::Point3f> > objectPoints;//3d vertices
    std::vector<std::vector<cv::Point2f> > imagePoints;//projection of 3d vertices

    //input stream
    VideoSourceLive videoSource;
    videoSource.resizeSource(0.5);
    
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    
    cv::Mat outputImage;
    int cpt_found_in_a_row = 0;
    
    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();
        
        bool found = false;
        std::vector<cv::Point2f> pointBuf;
        
        //process grid
        found = findCirclesGrid( inputImage, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID );
        
        if(found)
        {
            drawChessboardCorners( inputImage, boardSize, cv::Mat(pointBuf), found );
            cpt_found_in_a_row ++;
        }
        else
            cpt_found_in_a_row = 0;
        
        //plot counter
        char cStr[100];
        sprintf(cStr, "%lu", imagePoints.size());
        putText(inputImage, cStr,
                    cv::Point2i(inputImage.size().width - 40,inputImage.size().height-10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8, cvScalar(250,250,250), 1, CV_AA);

        imshow(window_name, inputImage);
        
        auto key = cv::waitKey(5);


        if(found && cpt_found_in_a_row == 20)//only add a detection when we had n=20 of them in a row
        //if(found && key == 32)//instead of waiting few detection in a row, wait for space key=> keep frame for calib and next frame
        {
            imagePoints.push_back(pointBuf);
            objectPoints.push_back(objectCorners);
            cpt_found_in_a_row = 0;
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

            //save to file, here don't save everything openCV usually saves in a calib file however it still respect same standard variable names
            cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
            cv::write(fs,"camera_matrix", cameraMatrix);
            cv::write(fs,"distortion_coefficients", distortionCoefficients);
            cv::write(fs,"image_width", inputImage.size().width);
            cv::write(fs,"image_height", inputImage.size().height);
            fs.release();
            
        }

        if(key == 27 || key == 'q')
            break;
    }
    return 0;
}

