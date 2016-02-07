
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace thymio_tracker
{

class LandmarkDetection;

class Landmark
{
public:
    static Landmark fromFile(const std::string& filename);
    
    Landmark(const cv::Mat& image,
             const std::vector<cv::KeyPoint>& keypoints,
             const cv::Mat& descriptors,
             const cv::Size2f& realSize);
    
    void find(const cv::Mat& image,
              const std::vector<cv::KeyPoint>& keypoints,
              const cv::Mat& descriptors,
              LandmarkDetection& detection) const;
    
    cv::Mat findHomography(const std::vector<cv::KeyPoint>& keypoints,
                            const cv::Mat& descriptors) const;
    
    // Get corners in template space
    std::vector<cv::Point2f> getCorners() const;
    
    inline const cv::Mat& getImage() const {return mImage;}
    
private:
    cv::Mat mImage;
    std::vector<cv::Mat> mPyramid;
    
    std::vector<cv::KeyPoint> mKeypoints;
    std::vector<cv::Point2f> mKeypointPos;
    cv::Mat mDescriptors;
    cv::Size2f mRealSize;
    
    cv::BFMatcher mMatcher;
};

class LandmarkDetection
{
    friend class Landmark;
    
public:
    LandmarkDetection(){}
    
    const cv::Mat& getHomography() const {return mHomography;}
    
private:
    cv::Mat mHomography;
};

}
