
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

void drawAxes(cv::Mat& image, const cv::Mat& orientation)
{
    static const cv::Scalar axes_colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    cv::Size::value_type width = image.size().width;
    cv::Point2d center(width * 0.1, width * 0.1);
    double length = width * 0.05;
    
    for(int i = 0; i < 3; ++i)
    {
        const cv::Point2d direction(orientation.at<float>(i, 1), orientation.at<float>(i, 0));
        const cv::Point2d arrow = center - length * direction;
        const cv::Scalar& color = axes_colors[i];
        
        cv::line(image, center, arrow, color);
    }
}

void loadCalibration(const std::string& filename,
                     IntrinsicCalibration* calibration,
                     const cv::Size* imgSize,
                     double scale = 1.0)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    readCalibrationFromFileStorage(fs, *calibration);
    
    cv::Size targetSize = *imgSize;
    targetSize.width *= scale;
    targetSize.height *= scale;
    rescaleCalibration(*calibration, targetSize);
    
    fs.release();
}

void init(const cv::Size &imgSize)
{
    // static const std::string ghfilename = "../data/GH_Arth_Perspective.dat";
    static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    mGH.loadFromFile(ghfilename);
    
    // loadCalibration("../data/calibration/embedded_camera_calib.xml", &calibration, &imgSize);
    // loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize);
    loadCalibration("/sdcard/nexus_camera_calib.xml", &calibration, &imgSize, mScale);
    mGH.setCalibration(calibration);
    
    initialized = true;
}

int process(const cv::Mat& input, cv::Mat& output,
            const cv::Mat* deviceOrientation)
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
    
    if(deviceOrientation)
        drawAxes(output, *deviceOrientation);
    
    return 0;
}
