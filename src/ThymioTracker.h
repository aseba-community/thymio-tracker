
#ifndef _THYMIOTRACKER_H
#define _THYMIOTRACKER_H

#include <string>
#include <sstream>
#include <ctime>



#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
// #include <opencv2/xfeatures2d.hpp>

#include "Landmark.hpp"
#include "Robot.hpp"

namespace thymio_tracker
{

struct DetectionInfo
{
    DetectionInfo(int numberOfLandmarks)
        : landmarkDetections(numberOfLandmarks)
    {}
    
    //robot detection info
    RobotDetection mRobotDetection;

    // Landmark detection information
    std::vector<LandmarkDetection> landmarkDetections;
    
    // Previous image
    cv::Mat prevImage;
};

struct CalibrationInfo
{
    CalibrationInfo(){};

    // Sets of matches
    std::vector<std::vector<cv::Point3f> > objectPoints;//3d vertices
    std::vector<std::vector<cv::Point2f> > imagePoints;//projection of 3d vertices

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
                  const std::string& externalFolder,
                  const std::vector<std::string>& landmarkFiles={});
    ThymioTracker(cv::FileStorage& calibration,
                  cv::FileStorage& geomHashing,
                  cv::FileStorage& robotModel,
                  std::vector<cv::FileStorage>& landmarkStorages);
    ~ThymioTracker(){}
    
    //standard function called by java wrapper
    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);

    //use detection information on some frames to update the calibration
    void calibrateOnline();

    
    //void drawLastDetection(cv::Mat* output) const;
    void drawLastDetection(cv::Mat* output, cv::Mat* deviceOrientation=0) const;
    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}
    inline const std::vector<Landmark>& getLandmarks() const {return mLandmarks;}
    
    inline const Timer& getTimer() const {return mTimer;}
    inline const IntrinsicCalibration& getIntrinsicCalibration() const {return mCalibration;}

private:
    void init(cv::FileStorage& calibration,
              cv::FileStorage& geomHashing,
              cv::FileStorage& robotModel,
              std::vector<cv::FileStorage>& landmarkStorages);

    /// Resize the calibration for a new given image size.
    void resizeCalibration(const cv::Size& imgSize);
    
    IntrinsicCalibration mCalibration;
    
    CalibrationInfo mCalibrationInfo;
    DetectionInfo mDetectionInfo;

    Robot mRobot;
    
    std::vector<Landmark> mLandmarks;
    cv::Ptr<cv::Feature2D> mFeatureExtractor;//want to extract features from current image once => put it out of landmark
    
    Timer mTimer;
    // cv::Ptr<cv::xfeatures2d::DAISY> mFeatureExtractor;
};

}

#endif // _THYMIOTRACKER_H
