
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace thymio_tracker
{

class Landmark
{
public:
    Landmark(const std::vector<cv::KeyPoint>& keypoints,
             const cv::Mat& descriptors,
             const cv::Size2i& imageSize,
             const cv::Size2f& realSize)
        : mKeypoints(keypoints)
        , mDescriptors(descriptors)
        , mImageSize(imageSize)
        , mRealSize(realSize)
        , mMatcher(cv::NORM_HAMMING)
    {}
    
    Landmark(const std::string& filename);
    
    cv::Mat findHomography(const std::vector<cv::KeyPoint>& keypoints,
                            const cv::Mat& descriptors) const;
    
    // Get corners in template space
    std::vector<cv::Point2f> getCorners() const;
    
private:
    std::vector<cv::KeyPoint> mKeypoints;
    cv::Mat mDescriptors;
    cv::Size2i mImageSize;
    cv::Size2f mRealSize;
    
    cv::BFMatcher mMatcher;
};

}
