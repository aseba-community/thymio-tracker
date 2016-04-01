
#ifndef _THYMIOTRACKER_H
#define _THYMIOTRACKER_H

#include <string>
#include <sstream>
#include <ctime>



#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
// #include <opencv2/xfeatures2d.hpp>

#include "GHscale.hpp"
#include "Models.hpp"
#include "Grouping.hpp"
#include "Landmark.hpp"

namespace thymio_tracker
{

struct DetectionInfo
{
    DetectionInfo(int numberOfLandmarks)
        : robotFound(false)
        , landmarkDetections(numberOfLandmarks)
    {}
    
    // Robot pose
    bool robotFound;
    cv::Affine3d robotPose;

    // Robot s tracking info
    cv::Mat mRobotHomography;
    std::map<int, cv::Point2f> mRobotCorrespondences;//corresp between robots keypoints index and current image
    
    // Landmark homographies
    std::vector<LandmarkDetection> landmarkDetections;
    // std::vector<std::pair<const Landmark*, cv::Mat> > landmarkPoses;
    
    // Debug info
    std::vector<cv::KeyPoint> blobs;
    std::vector<BlobPair> blobPairs;
    std::vector<BlobTriplet> blobTriplets;
    std::vector<BlobQuadruplets> blobQuadriplets;
    std::vector<cv::KeyPoint> blobsinTriplets;
    std::vector<DetectionGH> matches;
    
    // Previous image
    cv::Mat prevImage;

    void clear();
};

struct CalibrationInfo
{
    CalibrationInfo(){};

    // Sets of matches
    std::vector<std::vector<cv::Point3f> > objectPoints;
    std::vector<std::vector<cv::Point2f> > imagePoints;

    void clear();
};


class Timer
{
public:
    static constexpr int N = 10;
    
    Timer();
    void tic();
    
    inline double getFps() const {return mFps;}
    
private:
    std::clock_t mTicks[N];
    int mIndex;
    double mFps;
};


class ThymioTracker
{
public:
    ThymioTracker(const std::string& calibrationFile,
                  const std::string& geomHashingFile,
                  const std::vector<std::string>& landmarkFiles={});
    ThymioTracker(cv::FileStorage& calibration,
                  std::istream& geomHashing,
                  std::vector<cv::FileStorage>& landmarkStorages);
    ~ThymioTracker(){}
    
    //standard function called by java wrapper
    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);

    
    //void drawLastDetection(cv::Mat* output) const;
    void drawLastDetection(cv::Mat* output, cv::Mat* deviceOrientation=0) const;
    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}
    inline const std::vector<Landmark>& getLandmarks() const {return mLandmarks;}
    
    inline const Timer& getTimer() const {return mTimer;}
    inline const IntrinsicCalibration& getIntrinsicCalibration() const {return mCalibration;}

private:
    void init(cv::FileStorage& calibration,
              std::istream& geomHashing,
              std::vector<cv::FileStorage>& landmarkStorages);

    /// Resize the calibration for a new given image size.
    void resizeCalibration(const cv::Size& imgSize);
    
    GHscale mGH;
    ThymioBlobModel mRobot;
    IntrinsicCalibration mCalibration;
    Grouping mGrouping;
    
    DetectionInfo mDetectionInfo;
    CalibrationInfo mCalibrationInfo;
    
    std::vector<Landmark> mLandmarks;
    cv::Ptr<cv::Feature2D> mFeatureExtractor;
    
    Timer mTimer;
    // cv::Ptr<cv::xfeatures2d::DAISY> mFeatureExtractor;
};

}

#endif // _THYMIOTRACKER_H
