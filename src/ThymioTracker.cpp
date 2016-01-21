
#include "ThymioTracker.h"

#include <vector>
#include <stdexcept>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace thymio_tracker
{

void DetectionInfo::clearBlobs()
{
    blobs.clear();
    blobPairs.clear();
    blobTriplets.clear();
    blobQuadriplets.clear();
    blobsinTriplets.clear();
    matches.clear();
}

void drawPointsAndIds(cv::Mat& inputImage, const std::vector<DetectionGH>& matches)
{
    //draw Id
    for(unsigned int i=0;i<matches.size();i++)
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

ThymioTracker::ThymioTracker(const std::string& calibrationFile,
                             const std::string& geomHashingFile)
{
    //static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    std::ifstream geomHashingStream(geomHashingFile, std::ios::in | std::ios::binary);
    if (!geomHashingStream.is_open())
    {
        std::cerr << "Could not open " << geomHashingFile << std::endl;
        throw std::runtime_error("GHscale::loadFromFile > File not found!");
    }
    
    // loadCalibration("../data/calibration/embedded_camera_calib.xml", &calibration, &imgSize);
    // loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize);
    cv::FileStorage calibrationStorage(calibrationFile, cv::FileStorage::READ);
    if(!calibrationStorage.isOpened())
    {
        std::cerr << "Could not open " << calibrationFile << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    init(calibrationStorage, geomHashingStream);
}

ThymioTracker::ThymioTracker(cv::FileStorage& calibrationStorage,
                             std::istream& geomHashingStream)
{
    init(calibrationStorage, geomHashingStream);
}

void ThymioTracker::init(cv::FileStorage& calibrationStorage,
                         std::istream& geomHashingStream)
{
    mGH.loadFromStream(geomHashingStream);
    readCalibrationFromFileStorage(calibrationStorage, mCalibration);
    mGH.setCalibration(mCalibration);
}

void ThymioTracker::resizeCalibration(const cv::Size& imgSize)
{
    // loadCalibration(mCalibrationFile, imgSize, &mCalibration);
    rescaleCalibration(mCalibration, imgSize);
    mGH.setCalibration(mCalibration);
}

void ThymioTracker::update(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{    
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());
    
    mDetectionInfo.clearBlobs();

    //get the pairs which are likely to belong to group of blobs from model
    mGrouping.getBlobsAndPairs(input,
                               mDetectionInfo.blobs,
                               mDetectionInfo.blobPairs);
    
    // get triplet by checking homography and inertia
    mGrouping.getTripletsFromPairs(mDetectionInfo.blobs,
                                   mDetectionInfo.blobPairs,
                                   mDetectionInfo.blobTriplets);
    
    //get only blobs found in triplets
    getBlobsInTriplets(mDetectionInfo.blobs,
                       mDetectionInfo.blobTriplets,
                       mDetectionInfo.blobsinTriplets);
    
    mGrouping.getQuadripletsFromTriplets(mDetectionInfo.blobTriplets,
                                         mDetectionInfo.blobQuadriplets);
    
    //extract blobs and identify which one fit model, return set of positions and Id
    mGH.getModelPointsFromImage(mDetectionInfo.blobsinTriplets, mDetectionInfo.matches);
    
    mDetectionInfo.robotFound = mRobot.getPose(mCalibration,
                                               mDetectionInfo.matches,
                                               mDetectionInfo.robotPose,
                                               mDetectionInfo.robotFound);
}

void ThymioTracker::drawLastDetection(cv::Mat* output) const
{
    // mDetectionInfo.image.copyTo(*output);
    
    if(mDetectionInfo.robotFound)
        mRobot.draw(*output, mCalibration, mDetectionInfo.robotPose);
    else
        putText(*output, "Lost",
                cv::Point2i(10,10),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0,0,250), 1, CV_AA);
    
    drawBlobPairs(*output, mDetectionInfo.blobs, mDetectionInfo.blobPairs);
    drawBlobTriplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobTriplets);
    drawBlobQuadruplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobQuadriplets);
    // // drawPointsAndIds(output, mDetectionInfo.matches);
    
    // // if(deviceOrientation)
    // //     drawAxes(*output, *deviceOrientation);
}

}
