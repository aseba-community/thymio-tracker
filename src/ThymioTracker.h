
#ifndef _THYMIOTRACKER_H
#define _THYMIOTRACKER_H

#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
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
    
    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);
    
    void drawLastDetection(cv::Mat* output) const;
    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}
    inline const std::vector<Landmark>& getLandmarks() const {return mLandmarks;}
    
    inline const Timer& getTimer() const {return mTimer;}

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
    
    std::vector<Landmark> mLandmarks;
    cv::Ptr<cv::Feature2D> mFeatureExtractor;
    
    Timer mTimer;
    // cv::Ptr<cv::xfeatures2d::DAISY> mFeatureExtractor;
};

}

#endif // _THYMIOTRACKER_H
