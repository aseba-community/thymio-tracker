
#include "arthymio.h"

#include "GH.hpp"
#include "Models.hpp"

#include <vector>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

static bool initialized = false;
static bool mFound = false;
static GH mGH;
static ArthymioBlobModel mRobot;
static IntrinsicCalibration calibration;
static float mScale = 0.33;

void drawPointsAndIds(cv::Mat& inputImage, const std::vector<DetectionGH>& matches)
{
    //draw Id
    for(int i=0;i<matches.size();i++)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", matches[i].id);
        circle(inputImage, matches[i].position, 4, cvScalar(0,250,250), -1, 8, 0);
        putText(inputImage, pointIdStr, matches[i].position, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}

void loadCalibration(const std::string& filename, IntrinsicCalibration* calibration, cv::Size* imgSize, float scale = 1.0)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    fs["camera_matrix"] >> calibration->cameraMatrix;
    fs["distortion_coefficients"] >> calibration->distCoeffs;
    fs["image_width"] >> imgSize->width;
    fs["image_height"] >> imgSize->height;
    
    cv::Size sourceSize = *imgSize;
    imgSize->width *= scale;
    imgSize->height *= scale;
    resizeCameraMatrix(calibration->cameraMatrix, sourceSize, *imgSize);
}

void init()
{
     static const std::string ghfilename = "../data/GH_Arth_Perspective.dat";
    //static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    mGH.loadFromFile(ghfilename);
    cv::Size imgSize;
    
     loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize, mScale);
    ////loadCalibration("/sdcard/nexus_camera_calib.xml", &calibration, &imgSize, mScale);
    
    initialized = true;
}

int process(const cv::Mat& input, cv::Mat& output)
{
    if(!initialized)
        init();
    
    // input.copyTo(output);
    cv::resize(input, output, cv::Size(0, 0), mScale, mScale);
    
    cv::Affine3d robotPose;
    std::vector<DetectionGH> matches;
    mGH.getModelPointsFromImage(output, matches);
    
    mFound = mRobot.getPose(calibration, matches, robotPose, !mFound);
    if(mFound)
        mRobot.draw(output, calibration, robotPose);
    
    drawPointsAndIds(output, matches);
    
    return 0;
}

// int get_rows(const cv::Mat& m)
// {
//     return m.rows;
// }
