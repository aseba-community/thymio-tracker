
#include "ThymioTracker.h"

#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace thymio_tracker
{

void DetectionInfo::clear()
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
    for(unsigned int i=0; i<matches.size(); i++)
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
                     IntrinsicCalibration* calibration)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    readCalibrationFromFileStorage(fs, *calibration);
    fs.release();
}

ThymioTracker::ThymioTracker(const std::string& calibrationFile,
                             const std::string& geomHashingFile,
                             const std::vector<std::string>& landmarkFiles)
    : mCalibrationFile(calibrationFile)
    , mGeomHashingFile(geomHashingFile)
    , mDetectionInfo(landmarkFiles.size())
    , mFeatureExtractor(cv::ORB::create(1000))
{
    mGH.loadFromFile(mGeomHashingFile);
    loadCalibration(mCalibrationFile, &mCalibration);
    mGH.setCalibration(mCalibration);
    
    // Load landmarks
    for(auto landmarkFile : landmarkFiles)
        mLandmarks.push_back(Landmark(landmarkFile));
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
    
    // mDetectionInfo.clear();

    // //get the pairs which are likely to belong to group of blobs from model
    // mGrouping.getBlobsAndPairs(input,
    //                            mDetectionInfo.blobs,
    //                            mDetectionInfo.blobPairs);
    
    // // get triplet by checking homography and inertia
    // mGrouping.getTripletsFromPairs(mDetectionInfo.blobs,
    //                                mDetectionInfo.blobPairs,
    //                                mDetectionInfo.blobTriplets);
    
    // //get only blobs found in triplets
    // getBlobsInTriplets(mDetectionInfo.blobs,
    //                    mDetectionInfo.blobTriplets,
    //                    mDetectionInfo.blobsinTriplets);
    
    // mGrouping.getQuadripletsFromTriplets(mDetectionInfo.blobTriplets,
    //                                      mDetectionInfo.blobQuadriplets);
    
    // //extract blobs and identify which one fit model, return set of positions and Id
    // mGH.getModelPointsFromImage(mDetectionInfo.blobsinTriplets, mDetectionInfo.matches);
    
    // mDetectionInfo.robotFound = mRobot.getPose(mCalibration,
    //                                            mDetectionInfo.matches,
    //                                            mDetectionInfo.robotPose,
    //                                            mDetectionInfo.robotFound);
    
    // Landmark tracking
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat detectedDescriptors;
    mFeatureExtractor->detectAndCompute(input, cv::noArray(),
                                        detectedKeypoints, detectedDescriptors);
    
    auto landmarksIt = mLandmarks.cbegin();
    auto homographiesIt = mDetectionInfo.homographies.begin();
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++homographiesIt)
        *homographiesIt = landmarksIt->findHomography(detectedKeypoints, detectedDescriptors);
}

void ThymioTracker::drawLastDetection(cv::Mat* output) const
{
    // mDetectionInfo.image.copyTo(*output);
    
    // cv::drawMatches(*output, mDetectedKeypoints,
    //                     mLandmark.image, mLandmark.keypoints,
    //                     mMatches, *output,
    //                     cv::Scalar(0, 0, 255),
    //                     cv::Scalar(255, 0, 0),
    //                     std::vector<std::vector<char> >(),
    //                     cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    
    // if(mDetectionInfo.robotFound)
    //     mRobot.draw(*output, mCalibration, mDetectionInfo.robotPose);
    // else
    //     putText(*output, "Lost",
    //             cv::Point2i(10,10),
    //             cv::FONT_HERSHEY_COMPLEX_SMALL,
    //             0.8, cvScalar(0,0,250), 1, CV_AA);
    
    // drawBlobPairs(*output, mDetectionInfo.blobs, mDetectionInfo.blobPairs);
    // drawBlobTriplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobTriplets);
    // drawBlobQuadruplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobQuadriplets);
    // // drawPointsAndIds(output, mDetectionInfo.matches);
    
    // // if(deviceOrientation)
    // //     drawAxes(*output, *deviceOrientation);
    
    // Draw landmark detections
    std::vector<cv::Point2f> corners(4);
    
    auto homographiesIt = mDetectionInfo.homographies.cbegin();
    auto landmarksIt = mLandmarks.cbegin();
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++homographiesIt)
    {
        const Landmark& landmark = *landmarksIt;
        const cv::Mat& h = *homographiesIt;
        
        if(h.empty())
            continue;
        
        cv::perspectiveTransform(landmark.getCorners(), corners, h);
        cv::line(*output, corners[0], corners[1], cv::Scalar(0, 255, 0), 2);
        cv::line(*output, corners[1], corners[2], cv::Scalar(0, 255, 0), 2);
        cv::line(*output, corners[2], corners[3], cv::Scalar(0, 255, 0), 2);
        cv::line(*output, corners[3], corners[0], cv::Scalar(0, 255, 0), 2);
    }
}

}
