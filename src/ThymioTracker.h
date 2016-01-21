
#ifndef _THYMIOTRACKER_H
#define _THYMIOTRACKER_H

#include <opencv2/core.hpp>

#include "GHscale.hpp"
#include "Models.hpp"
#include "Grouping.hpp"

namespace thymio_tracker
{

struct DetectionInfo
{
    DetectionInfo()
        : robotFound(false)
    {}
    
    bool robotFound;
    cv::Affine3d robotPose;
    
    std::vector<cv::KeyPoint> blobs;
    std::vector<BlobPair> blobPairs;
    std::vector<BlobTriplet> blobTriplets;
    std::vector<BlobQuadruplets> blobQuadriplets;
    std::vector<cv::KeyPoint> blobsinTriplets;
    std::vector<DetectionGH> matches;

    void clearBlobs();
};

class ThymioTracker
{
public:
    ThymioTracker(const std::string& calibrationFile,
                  const std::string& geomHashingFile);
    ThymioTracker(cv::FileStorage& calibration,
                  std::istream& geomHashing);
    ~ThymioTracker(){}
    
    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);
    
    void drawLastDetection(cv::Mat* output) const;
    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}

private:
    void init(cv::FileStorage& calibration,
              std::istream& geomHashing);

    /// Resize the calibration for a new given image size.
    void resizeCalibration(const cv::Size& imgSize);
    
    GHscale mGH;
    ThymioBlobModel mRobot;
    IntrinsicCalibration mCalibration;
    Grouping mGrouping;
    
    DetectionInfo mDetectionInfo;
};

}

#endif // _THYMIOTRACKER_H
