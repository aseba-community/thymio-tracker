
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
static float mScale = 1.0;

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

void loadCalibration(const std::string& filename, IntrinsicCalibration* calibration, const cv::Size* imgSize)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    readCalibrationFromFileStorage(fs,*calibration);
    rescaleCalibration(*calibration, *imgSize);
    
    fs.release();
}

void init(const cv::Size &imgSize)
{
     static const std::string ghfilename = "../data/GH_Arth_Perspective.dat";
    //static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    mGH.loadFromFile(ghfilename);
    
    loadCalibration("../data/calibration/embedded_camera_calib.xml", &calibration, &imgSize);
    //loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize);
    ////loadCalibration("/sdcard/nexus_camera_calib.xml", &calibration, &imgSize, mScale);
    mGH.setCalibration(calibration);
    
    initialized = true;
}

int process(const cv::Mat& input, cv::Mat& output)
{
    // input.copyTo(output);
    cv::resize(input, output, cv::Size(0, 0), mScale, mScale);

    if(!initialized)
        init(output.size());
    
    
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
