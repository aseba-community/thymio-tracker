
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
    DetectionInfo(){};
    DetectionInfo(int numberOfLandmarks)
        : landmarkDetections(numberOfLandmarks){}

    void init(int numberOfLandmarks){landmarkDetections.resize(numberOfLandmarks);}
    
    //robot detection info
    RobotDetection mRobotDetection;

    // Landmark detection information
    std::vector<LandmarkDetection> landmarkDetections;
    
    // Previous image (as robot and landmark detection might run on separate threads,
    //each need to store its previous frame)
    cv::Mat prevImageRobot;
    cv::Mat prevImageLandm;
};

struct CalibrationInfo
{
    CalibrationInfo(){nbFramesForCalibration = 10;};

    // number of frames to acquires till we can calibrate (0 = calibration is done)
    unsigned int getNbFramesToCalibration() const {return nbFramesForCalibration>objectPoints.size() ? nbFramesForCalibration-objectPoints.size() : 0;}

    float getProgress() const { return float(objectPoints.size()) / float(nbFramesForCalibration); }

    unsigned int nbFramesForCalibration;
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
    ThymioTracker(const std::string& configPath);
    ThymioTracker(const std::string& calibrationFile,
                  const std::string& externalFolder,
                  const std::vector<std::string>& landmarkFiles={});
    ThymioTracker(cv::FileStorage& calibration,
                  cv::FileStorage& geomHashing,
                  cv::FileStorage& robotModel,
                  std::vector<cv::FileStorage>& landmarkStorages);
    ~ThymioTracker(){}
    
    //detection information updates, => use that to perform calibration
    bool updateCalibration();
    void writeCalibration(cv::FileStorage& output);

    //to detect and track the robot
    void updateRobot(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);

    //to detect and track the landmarks
    void updateLandmarks(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0);

    void update(const cv::Mat& input,
                const cv::Mat* deviceOrientation=0){updateRobot(input,deviceOrientation);updateLandmarks(input,deviceOrientation);};
    //void update(const cv::Mat& input,
    //            const cv::Mat* deviceOrientation=0){updateLandmarks(input,deviceOrientation);};

    void drawLastDetection(cv::Mat* output, cv::Mat* deviceOrientation=0) const;

    
    inline const DetectionInfo& getDetectionInfo() const {return mDetectionInfo;}
    inline const CalibrationInfo& getCalibrationInfo() const {return mCalibrationInfo;}
    inline const std::vector<Landmark>& getLandmarks() const {return mLandmarks;}
    
    inline const Timer& getTimer() const {return mTimer;}
    inline const IntrinsicCalibration& getIntrinsicCalibration() const {return mCalibration;}

private:
    void init(const std::string& calibration,
              const std::string& geomHashing,
              const std::string& robotModel,
              const std::vector<std::string>& landmarkStorages);
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
