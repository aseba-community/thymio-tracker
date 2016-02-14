
#include "Landmark.hpp"

#include <stdexcept>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <iostream>

namespace thymio_tracker
{

Landmark Landmark::fromFile(const std::string& filename)
{
    cv::Mat image;
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // cv::Size2f realSize;
    std::vector<float> realSize;
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    if(!fs.isOpened())
        throw std::runtime_error("Marker file not found");
    
    cv::read(fs["keypoints"], keypoints);
    cv::read(fs["descriptors"], descriptors);
    // cv::read(fs["image_size"], imageSize);
    cv::read(fs["real_size"], realSize);
    cv::read(fs["image"], image);
    fs.release();
    
    if(image.empty())
        throw std::runtime_error("Could not load image data");
    
    return Landmark(image, keypoints, descriptors, cv::Size(realSize[0], realSize[1]));
}

Landmark::Landmark(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const cv::Mat& descriptors,
                    const cv::Size2f& realSize)
    : mImage(image)
    , mKeypoints(keypoints)
    , mKeypointPos(keypoints.size())
    , mDescriptors(descriptors)
    , mRealSize(realSize)
    , mMatcher(cv::NORM_HAMMING)
{
    cv::buildOpticalFlowPyramid(image, mPyramid, cv::Size(21, 21), 0);
    
    std::transform(mKeypoints.begin(), mKeypoints.end(), mKeypointPos.begin(),
        [](const cv::KeyPoint& kp){return kp.pt;});
}

void Landmark::find(const cv::Mat& image,
              const cv::Mat& prevImage,
              const std::vector<cv::KeyPoint>& keypoints,
              const cv::Mat& descriptors,
              LandmarkDetection& detection) const
{
    std::vector<cv::Point2f> scenePoints;
    std::vector<int> correspondences;
    
    this->findCorrespondencesWithTracking(image, prevImage, detection, scenePoints, correspondences);
    
    if(!keypoints.empty())
        this->findCorrespondencesWithKeypoints(keypoints, descriptors, scenePoints, correspondences);
    // else
    //     this->findCorrespondencesWithTracking(image, prevImage, detection, scenePoints, correspondences);
    
    std::vector<cv::Point2f> objectPoints;
    for(int c : correspondences)
        objectPoints.push_back(mKeypoints[c].pt);
    
    // Compute homography
    cv::Mat homography;
    std::vector<unsigned char> mask;
    if(!scenePoints.empty())
        homography = cv::findHomography(objectPoints, scenePoints, CV_RANSAC, 3, mask);
    
    // Save homography and inliers
    detection.mHomography = homography;
    
    detection.mCorrespondences.clear();
    auto maskIt = mask.cbegin();
    auto correspIt = correspondences.cbegin();
    auto scenePointsIt = scenePoints.cbegin();
    for(; maskIt != mask.cend(); ++maskIt, ++correspIt, ++scenePointsIt)
    {
        if(!*maskIt)
            continue;
        
        detection.mCorrespondences[*correspIt] = *scenePointsIt;
    }
}

void Landmark::findCorrespondencesWithKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                const cv::Mat& descriptors,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    std::vector<std::vector<cv::DMatch> > matches;
    
    mMatcher.knnMatch(descriptors, mDescriptors, matches, 2);
    
    // Keep only significant matches
    std::vector<cv::DMatch> goodMatches;
    for(auto el : matches)
    {
        if(el[0].distance < 0.75 * el[1].distance)
            goodMatches.push_back(el[0]);
    }
    
    for(auto match : goodMatches)
    {
        correspondences.push_back(match.trainIdx);
        scenePoints.push_back(keypoints[match.queryIdx].pt);
    }
}

void Landmark::findCorrespondencesWithTracking(const cv::Mat& image,
                                const cv::Mat& prevImage,
                                const LandmarkDetection& prevDetection,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    if(prevImage.empty() || prevDetection.mCorrespondences.empty())
        return;
    
    // Get positions of keypoints in previous frame
    std::vector<cv::Point2f> prevPoints;
    for(auto p : prevDetection.mCorrespondences)
        prevPoints.push_back(p.second);
    
    // Optical flow
    std::vector<cv::Mat> pyramid1, pyramid2;
    buildOpticalFlowPyramid(image, pyramid1, cv::Size(21, 21), 2);
    buildOpticalFlowPyramid(prevImage, pyramid2, cv::Size(21, 21), 2);
    std::vector<cv::Point2f> nextPoints;
    std::vector<unsigned char> status;
    // cv::calcOpticalFlowPyrLK(prevImage, image, prevPoints, nextPoints, status, cv::noArray(), cv::Size(8, 8));
    cv::calcOpticalFlowPyrLK(pyramid2, pyramid1, prevPoints, nextPoints, status, cv::noArray(), cv::Size(8, 8));
    
    // Keep only found keypoints
    auto statusIt = status.cbegin();
    auto nextPointsIt = nextPoints.cbegin();
    auto correspIt = prevDetection.mCorrespondences.cbegin();
    for(; statusIt != status.cend(); ++statusIt, ++nextPointsIt, ++correspIt)
    {
        if(!*statusIt)
            continue;
        
        scenePoints.push_back(*nextPointsIt);
        correspondences.push_back(correspIt->first);
    }
}

cv::Mat Landmark::findHomography(const std::vector<cv::KeyPoint>& keypoints,
                                    const cv::Mat& descriptors) const
{
    std::vector<std::vector<cv::DMatch> > matches;
    
    mMatcher.knnMatch(descriptors, mDescriptors, matches, 2);
    
    // Keep only significant matches
    std::vector<cv::DMatch> goodMatches;
    for(auto el : matches)
    {
        if(el[0].distance < 0.75 * el[1].distance)
            goodMatches.push_back(el[0]);
    }
    
    // Fit a homography.
    std::vector<cv::Point2f> object_points;
    std::vector<cv::Point2f> scenePoints;
    for(auto match : goodMatches)
    {
        object_points.push_back(mKeypoints[match.trainIdx].pt);
        scenePoints.push_back(keypoints[match.queryIdx].pt);
    }
    
    cv::Mat homography;
    if(!scenePoints.empty())
        homography = cv::findHomography(object_points, scenePoints, CV_RANSAC);
    
    return homography;
}

std::vector<cv::Point2f> Landmark::getCorners() const
{
    std::vector<cv::Point2f> res(4);
    
    const cv::Size size = mImage.size();
    
    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(size.width, 0);
    res[2] = cv::Point2f(size.width, size.height);
    res[3] = cv::Point2f(0, size.height);
    
    return res;
}

}
