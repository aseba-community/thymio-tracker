
#include "Calibrator.hpp"

#include <string>
#include <iostream>
#include <fstream>

#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace thymio_tracker
{


Calibrator::Calibrator(const std::string& calibFileName)
{
    fs.open(calibFileName, cv::FileStorage::WRITE);

    //std::ofstream myfile;
    //myfile.open (calibFileName);
    //myfile << "Writing this to a file.\n";
    //myfile.close();      

    init(&fs);

}

Calibrator::Calibrator(cv::FileStorage* calibStorage)
{
    init(calibStorage);
}

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners)
{
    corners.clear();

    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            corners.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
}


void Calibrator::init(cv::FileStorage* calibStorage)
{
    fsPtr = calibStorage;

    calibrated = false;
    cpt_found_in_a_row = 0;
    nb_required_images = 10;

    cv::Size boardSize(4,11);
    calcBoardCornerPositions(boardSize, 0.02, objectCorners);

}

void Calibrator::update(const cv::Mat& inputImage)
{    

    found = false;
    pointBuf.clear();
    
    //process grid
    cv::Size boardSize(4,11);
    found = cv::findCirclesGrid( inputImage, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID );
    
    if(!calibrated)
    {

    if(found)
        cpt_found_in_a_row ++;
    else
        cpt_found_in_a_row = 0;

    //if(found && key == 32)//space => keep frame for calib and next frame
    if(found && cpt_found_in_a_row == 20)//only add a detection when we had n of them in a row
    {
        imagePoints.push_back(pointBuf);
        objectPoints.push_back(objectCorners);
        cpt_found_in_a_row = 0;
    }

    if(imagePoints.size() == nb_required_images)
    {
        std::vector<cv::Mat> rotationVectors;
        std::vector<cv::Mat> translationVectors;

        cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

        int flags = 0;
        double rms = cv::calibrateCamera(objectPoints, imagePoints, inputImage.size(), cameraMatrix,
                      distortionCoefficients, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

        std::cout<<"camera calibration done ... RMS = "<<rms<<std::endl;

        //save to file
        cv::write(*fsPtr,"camera_matrix", cameraMatrix);
        cv::write(*fsPtr,"distortion_coefficients", distortionCoefficients);
        cv::write(*fsPtr,"image_width", inputImage.size().width);
        cv::write(*fsPtr,"image_height", inputImage.size().height);
        fsPtr->release();        

        calibrated = true;
    }
}

}

void Calibrator::drawState(cv::Mat* output) const
{
    cv::Size boardSize(4,11);

    if(found)
        drawChessboardCorners( *output, boardSize, cv::Mat(pointBuf), found );

    //plot counter
    if(calibrated)
    {
        putText(*output, "Calibrated !",
                    cv::Point2i(output->size().width/2,output->size().height/2),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    2., cvScalar(250,250,250), 1, CV_AA); 
    }
    else
    {
        char cStr[100];
        sprintf(cStr, "Nb frame to acquire: %lu", nb_required_images-imagePoints.size());
        putText(*output, cStr,
                    cv::Point2i(20,20),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8, cvScalar(250,250,250), 1, CV_AA);
    }


}

}
