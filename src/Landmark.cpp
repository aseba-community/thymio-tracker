
#include "Landmark.hpp"

#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace thymio_tracker
{

Landmark::Landmark(const std::string& filename)
    : mMatcher(cv::NORM_HAMMING)
    // : mMatcher(cv::NORM_L2)
{
    std::vector<int> imageSize;
    std::vector<float> realSize;
    
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    if(!fs.isOpened())
        throw std::runtime_error("Marker file not found!");
    
    cv::read(fs["keypoints"], mKeypoints);
    cv::read(fs["descriptors"], mDescriptors);
    cv::read(fs["image_size"], imageSize);
    cv::read(fs["real_size"], realSize);
    fs.release();
    
    mImageSize = cv::Size(imageSize[0], imageSize[1]);
    mRealSize = cv::Size(realSize[0], realSize[1]);
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
    
    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(mImageSize.width, 0);
    res[2] = cv::Point2f(mImageSize.width, mImageSize.height);
    res[3] = cv::Point2f(0, mImageSize.height);
    
    return res;
}

}
