
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <map>
#include "Generic.hpp"

namespace thymio_tracker
{

class LandmarkDetection;

class Landmark
{
public:
    static Landmark fromFileStorage(cv::FileStorage& fs);
    
    Landmark(const cv::Mat& image,
             const std::vector<cv::KeyPoint>& keypoints,
             const cv::Mat& descriptors,
             const cv::Size2f& realSize);
    
    void find(const cv::Mat& image,
              const cv::Mat& prevImage,
              const IntrinsicCalibration& mCalibration,
              const std::vector<cv::KeyPoint>& keypoints,
              const cv::Mat& descriptors,
              LandmarkDetection& detection) const;
    
    void findCorrespondencesWithKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                const cv::Mat& descriptors,
                                std::vector<cv::Point2f>& scene_points,
                                std::vector<int>& correspondences) const;
    
    void findCorrespondencesWithActiveSearch(const cv::Mat& image,
                                const LandmarkDetection& prevDetection,
                                std::vector<cv::Point2f>& scene_points,
                                std::vector<int>& correspondences) const;
    
    void findCorrespondencesWithTracking(const cv::Mat& image,
                                const cv::Mat& prevImage,
                                const LandmarkDetection& prevDetection,
                                std::vector<cv::Point2f>& scene_points,
                                std::vector<int>& correspondences) const;
    
    cv::Mat findHomography(const std::vector<cv::KeyPoint>& keypoints,
                            const cv::Mat& descriptors) const;
    
    // Get corners in template space
    std::vector<cv::Point2f> getCorners() const;

    inline const cv::Size2f getRealSize() const {return mRealSize;};
    inline const cv::Mat& getImage() const {return mImage;}
    const std::vector<cv::Point2f>& getKeypointPos() const {return mKeypointPos;}
    
private:
    cv::Mat mImage;
    std::vector<cv::Mat> mPyramid;
    
    std::vector<cv::KeyPoint> mKeypoints;
    std::vector<cv::Point2f> mKeypointPos;
    cv::Mat mDescriptors;
    cv::Size2f mRealSize;//in cm
    
    cv::BFMatcher mMatcher;
};

class LandmarkDetection
{
    friend class Landmark;
    
public:
    LandmarkDetection(){}
    
    const cv::Mat& getHomography() const {return mHomography;}
    const cv::Affine3d& getPose() const {return mPose;}
    const std::map<int, cv::Point2f>& getCorrespondences() const {return mCorrespondences;}
    bool isFound() const {return !mHomography.empty();}
    
private:
    cv::Mat mHomography;
    cv::Affine3d mPose;
    
    std::map<int, cv::Point2f> mCorrespondences;
    // std::vector<cv::Point2f> mInliers;
};

}
