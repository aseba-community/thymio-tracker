
#ifndef _THYMIOTRACKER_H
#define _THYMIOTRACKER_H

#include <opencv2/core.hpp>

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
        , homographies(numberOfLandmarks)
    {}
    
    // Robot pose
    bool robotFound;
    cv::Affine3d robotPose;
    
    // Landmark homographies
    std::vector<cv::Mat> homographies;
    // std::vector<std::pair<const Landmark*, cv::Mat> > landmarkPoses;
    
    // Debug info
    std::vector<cv::KeyPoint> blobs;
    std::vector<BlobPair> blobPairs;
    std::vector<BlobTriplet> blobTriplets;
    std::vector<BlobQuadruplets> blobQuadriplets;
    std::vector<cv::KeyPoint> blobsinTriplets;
    std::vector<DetectionGH> matches;

    void clear();
};

class ThymioTracker
{
public:
    ThymioTracker(const std::string& calibrationFile,
                  const std::string& geomHashingFile,
                  const std::vector<std::string>& landmarkFiles={});
    ~ThymioTracker(){}
    
    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);
    
    void drawLastDetection(cv::Mat* output) const;
    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}

private:
    /// Resize the calibration for a new given image size.
    void resizeCalibration(const cv::Size& imgSize);
    
    const std::string mCalibrationFile;
    const std::string mGeomHashingFile;
    GHscale mGH;
    ThymioBlobModel mRobot;
    IntrinsicCalibration mCalibration;
    Grouping mGrouping;
    
    DetectionInfo mDetectionInfo;
    
    std::vector<Landmark> mLandmarks;
    cv::Ptr<cv::ORB> mFeatureExtractor;
};

}

#endif // _THYMIOTRACKER_H
