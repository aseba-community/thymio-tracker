
#include "Landmark.hpp"

#include <stdexcept>

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
              const std::vector<cv::KeyPoint>& keypoints,
              const cv::Mat& descriptors,
              LandmarkDetection& detection) const
{
    if(!keypoints.empty())
        detection.mHomography = this->findHomography(keypoints, descriptors);
    
    // TODO: Remove this
    if(detection.mHomography.empty())
        return;
    
    const cv::Mat& h = detection.mHomography;
    cv::Mat crop;
    std::vector<cv::Mat> cropPyramid;
    cv::warpPerspective(image, crop, h.inv(), mImage.size());
    // cv::buildOpticalFlowPyramid(crop, cropPyramid, cv::Size(21, 21), 0);
    
    std::vector<cv::Point2f> nextPts;
    std::vector<unsigned char> status;
    // std::cout << "Tracking " << mKeypoints.size() << " keypoints in a " 
    //           << crop.size() << " image." << std::endl;
    cv::calcOpticalFlowPyrLK(mPyramid, crop, mKeypointPos, nextPts, status, cv::noArray(),
                            cv::Size(15, 15));
    
    // TODO: Filter points according to status.
    
    cv::perspectiveTransform(nextPts, nextPts, h);
    detection.mHomography = cv::findHomography(mKeypointPos, nextPts, CV_RANSAC);
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
    std::vector<cv::Point2f> scene_points;
    for(auto match : goodMatches)
    {
        object_points.push_back(mKeypoints[match.trainIdx].pt);
        scene_points.push_back(keypoints[match.queryIdx].pt);
    }
    
    cv::Mat homography;
    if(!scene_points.empty())
        homography = cv::findHomography(object_points, scene_points, CV_RANSAC);
    
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
