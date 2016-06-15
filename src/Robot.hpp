
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <map>
#include "GHscale.hpp"
#include "Models.hpp"
#include "Grouping.hpp"


namespace thymio_tracker
{

class RobotDetection;

class Robot
{
public:
    Robot(){};

    void init(IntrinsicCalibration *_mCalibration_ptr,
              cv::FileStorage& geomHashingStorage,
              cv::FileStorage& robotModelStorage);
    
    void find(const cv::Mat& image,
              const cv::Mat& prevImage,
              RobotDetection& detection) const;

    void findFromBlobGroupsAndGH(const cv::Mat& image,
                                 RobotDetection& detection) const;

    /*void findCorrespondencesWithTracking(const cv::Mat& image,
                                const cv::Mat& prevImage,
                                const RobotDetection& prevDetection,
                                std::vector<cv::Point2f>& scene_points,
                                std::vector<int>& correspondences) const;

    void findCorrespondencesWithActiveSearch(const cv::Mat& image,
                                const RobotDetection& prevDetection,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const;*/

    
    //find homography from top view to current image
    
    const ThymioBlobModel& model() const {return mModel;}
    
private:
    //calibration
    IntrinsicCalibration *mCalibration_ptr;

    //for detection
    Grouping mGrouping;
    GHscale mGH;
    ThymioBlobModel mModel;
};

class RobotDetection
{
    friend class Robot;
    
public:
    RobotDetection()
        : robotFound(false)
        {}
    
    //const cv::Mat& getHomography() const {return mHomography;}
    const cv::Affine3d& getPose() const {return mPose;}
    //const std::map<int, cv::Point2f>& getCorrespondences() const {return mCorrespondences;}
    const bool& isFound() const {return robotFound;}

    void clearBlobs();
    void drawBlobs(cv::Mat* output) const;
    
protected:
    //output info
    bool robotFound;
    //cv::Mat mHomography;
    cv::Affine3d mPose;
    
    //temporal detection variables
    std::vector<cv::KeyPoint> blobs;
    std::vector<BlobPair> blobPairs;
    std::vector<BlobTriplet> blobTriplets;
    std::vector<BlobQuadruplets> blobQuadriplets;
    std::vector<cv::KeyPoint> blobsinTriplets;
    std::vector<DetectionGH> matches;

    //temporary tracking variables
    //std::map<int, cv::Point2f> mCorrespondences;
};

}
