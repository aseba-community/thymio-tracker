
#include "ThymioTracker.h"

#include <vector>
#include <stdexcept>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace thymio_tracker
{

static const std::vector<cv::Scalar> colorPalette = {
    cv::Scalar(76, 114, 176),
    cv::Scalar(85, 168, 104),
    cv::Scalar(196, 78, 82),
    cv::Scalar(129, 114, 178),
    cv::Scalar(204, 185, 116),
    cv::Scalar(100, 181, 205)
};

Timer::Timer()
    : mTicks{0}
    , mIndex(0)
    , mFps(-1.0)
{}

void Timer::tic()
{
    std::clock_t current = std::clock();
    std::clock_t prev = mTicks[mIndex];
    mTicks[mIndex] = current;
    ++mIndex;
    if(mIndex >= N)
        mIndex = 0;
    
    if(prev != 0)
        mFps = CLOCKS_PER_SEC * N / static_cast<double>(current - prev);
}

void CalibrationInfo::clear()
{
    objectPoints.clear();
    imagePoints.clear();
}

void drawPointsAndIds(cv::Mat& inputImage, const std::vector<DetectionGH>& matches)
{
    //draw Id
    for(unsigned int i = 0; i < matches.size(); ++i)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", matches[i].id);
        circle(inputImage, matches[i].position, 4, cvScalar(0, 250, 250), -1, 8, 0);
        putText(inputImage, pointIdStr, matches[i].position, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}

void drawAxes(cv::Mat& image, const cv::Mat& orientation)
{
    static const cv::Scalar axes_colors[] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    cv::Size::value_type width = image.size().width;
    //cv::Point2d center(width * 0.1, width * 0.1);
    cv::Point2d center(width * 0.5, image.size().height * 0.5);
    double length = width * 0.05;
    
    for(int i = 0; i < 3; ++i)
    {
        const cv::Point2d direction(orientation.at<float>(i, 1), orientation.at<float>(i, 0));
        const cv::Point2d arrow = center - length * direction;
        const cv::Scalar& color = axes_colors[i];
        
        cv::line(image, center, arrow, color);
    }
}

ThymioTracker::ThymioTracker(const std::string& calibrationFile,
                             const std::string& geomHashingFile,
                             const std::vector<std::string>& landmarkFiles)
    : mDetectionInfo(landmarkFiles.size())
    , mFeatureExtractor(cv::BRISK::create())
{
    //static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    /*std::ifstream geomHashingStream(geomHashingFile, std::ios::in | std::ios::binary);
    if (!geomHashingStream.is_open())
    {
        std::cerr << "Could not open " << geomHashingFile << std::endl;
        throw std::runtime_error("GHscale::loadFromFile > File not found!");
    }*/
    cv::FileStorage geomHashingStorage(geomHashingFile, cv::FileStorage::READ);
    if (!geomHashingStorage.isOpened())
    {
        std::cerr << "Could not open " << geomHashingFile << std::endl;
        throw std::runtime_error("GHscale::loadFromFile > File not found!");
    }
    
    // loadCalibration("../data/calibration/embedded_camera_calib.xml", &calibration, &imgSize);
    // loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize);
    cv::FileStorage calibrationStorage(calibrationFile, cv::FileStorage::READ);
    if(!calibrationStorage.isOpened())
    {
        std::cerr << "Could not open " << calibrationFile << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    std::vector<cv::FileStorage> landmarkStorages;
    for(auto& landmarkFile : landmarkFiles)
    {
        cv::FileStorage fs(landmarkFile, cv::FileStorage::READ);
        if(!fs.isOpened())
            throw std::runtime_error("Marker file not found");
        landmarkStorages.push_back(fs);
    }
    
    init(calibrationStorage, geomHashingStorage, landmarkStorages);
    //init(calibrationStorage, geomHashingStream, landmarkStorages);
    
}

ThymioTracker::ThymioTracker(cv::FileStorage& calibrationStorage,
                             cv::FileStorage& geomHashingStorage,
                             std::vector<cv::FileStorage>& landmarkStorages)
    : mDetectionInfo(landmarkStorages.size())
    , mFeatureExtractor(cv::BRISK::create())
{
    init(calibrationStorage, geomHashingStorage, landmarkStorages);
}

void ThymioTracker::init(cv::FileStorage& calibrationStorage,
                         cv::FileStorage& geomHashingStorage,
                         //std::istream& geomHashingStream,
                         std::vector<cv::FileStorage>& landmarkStorages)
{
    readCalibrationFromFileStorage(calibrationStorage, mCalibration);

    //mRobot.init(&mCalibration, geomHashingStream);
    mRobot.init(&mCalibration, geomHashingStorage);
    //mGH.loadFromStream(geomHashingStream);
    //mGH.setCalibration(mCalibration);
    
    // Load landmarks
    for(auto& landmarkStorage : landmarkStorages)
        mLandmarks.push_back(Landmark::fromFileStorage(landmarkStorage));
}

void ThymioTracker::resizeCalibration(const cv::Size& imgSize)
{
    // loadCalibration(mCalibrationFile, imgSize, &mCalibration);
    rescaleCalibration(mCalibration, imgSize);
}


    
void ThymioTracker::update(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{    
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());

    //to do some debugging and plot stuff on current image, 
    //need to copy frame first as it will be used to set previous frame for tracking
    cv::Mat cpImg;
    input.copyTo(cpImg);
    
    // Robot detection and tracking
    //mRobot.find(input,mDetectionInfo.prevImage,mDetectionInfo.mRobotDetection);

    // Landmark detection and tracking
    static int counter = 100;   
    ++counter;
    
    //check if all the landmarks are tracked
    bool allTracked = true;
    auto lmcDetectionsIt = mDetectionInfo.landmarkDetections.cbegin();
    for(; lmcDetectionsIt != mDetectionInfo.landmarkDetections.cend(); ++lmcDetectionsIt)
    {
        const cv::Mat& h = lmcDetectionsIt->getHomography();
        if(h.empty()) 
            allTracked = false;
    }


    // Extract features only once every 20 frames and only if need to do any detection (ie all markers are not tracked)
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat detectedDescriptors;
    if(!allTracked && counter >= 20)
    {
        mFeatureExtractor->detectAndCompute(input, cv::noArray(),
                                            detectedKeypoints, detectedDescriptors);
        counter = 0;
    }
    
    auto landmarksIt = mLandmarks.cbegin();
    auto lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
        landmarksIt->find(input, mDetectionInfo.prevImage, mCalibration, detectedKeypoints, detectedDescriptors, *lmDetectionsIt);
    
    //calibration of camera: if needed, capture a set of frames where tags are tracked and use them to calibrate the camera
    //add the matches to the calibration information
    //calibrateOnline();

    cpImg.copyTo(mDetectionInfo.prevImage);
    //input.copyTo(mDetectionInfo.prevImage);
    
    mTimer.tic();
}

void ThymioTracker::calibrateOnline()
{
    static int counter_calib = 100;
    int nb_detections_for_calibration = 10;
    ++counter_calib;

    if(mCalibrationInfo.objectPoints.size() < nb_detections_for_calibration && counter_calib >= 20)
    {
        //for each tracked landmark add the matches to the calibration tool
        //ie for each image and each landmark, the set of 3D points and their projections
        auto landmarksIt = mLandmarks.cbegin();
        auto lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();

        for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
        {
            const cv::Mat& h = lmDetectionsIt->getHomography();
            if(!h.empty() && lmDetectionsIt->getCorrespondences().size()>100)
            {
                std::vector<cv::Point3f> lmObjectPoints;
                std::vector<cv::Point2f> lmImagePoints;

                float scale = landmarksIt->getRealSize().width/landmarksIt->getImage().size().width;
                auto correspIt = lmDetectionsIt->getCorrespondences().cbegin();
                for(; correspIt != lmDetectionsIt->getCorrespondences().cend(); ++correspIt)
                {
                    lmImagePoints.push_back(correspIt->second);
                    cv::Point3f lmPoint = cv::Point3f(scale*landmarksIt->getKeypointPos()[correspIt->first].x,scale*landmarksIt->getKeypointPos()[correspIt->first].y,0.);
                    lmObjectPoints.push_back(lmPoint);
                }


                mCalibrationInfo.objectPoints.push_back(lmObjectPoints);
                mCalibrationInfo.imagePoints.push_back(lmImagePoints);

                //std::cout<<"nb matches for calib : "<<lmObjectPoints.size()<<std::endl;
            }
        }

        if(mCalibrationInfo.objectPoints.size() >= nb_detections_for_calibration)
        {
            std::vector<cv::Mat> rotationVectors;
            std::vector<cv::Mat> translationVectors;

            cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

            int flags = 0;
            double rms = calibrateCamera(mCalibrationInfo.objectPoints, mCalibrationInfo.imagePoints, mCalibration.imageSize, mCalibration.cameraMatrix,
                          mCalibration.distCoeffs, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

            std::cout<<"camera calibration RMS = "<<rms<<std::endl;
        }

        counter_calib = 0;
    }
}

void ThymioTracker::drawLastDetection(cv::Mat* output, cv::Mat* deviceOrientation) const
{
    // mDetectionInfo.image.copyTo(*output);
    
    // cv::drawMatches(*output, mDetectedKeypoints,
    //                     mLandmark.image, mLandmark.keypoints,
    //                     mMatches, *output,
    //                     cv::Scalar(0, 0, 255),
    //                     cv::Scalar(255, 0, 0),
    //                     std::vector<std::vector<char> >(),
    //                     cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    
    if(mDetectionInfo.mRobotDetection.isFound())
        mRobot.model().draw(*output, mCalibration, mDetectionInfo.mRobotDetection.getPose());
    else
        putText(*output, "Lost",
                cv::Point2i(10,10),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0,0,250), 1, CV_AA);
    
    //drawBlobPairs(*output, mDetectionInfo.blobs, mDetectionInfo.blobPairs);
    //drawBlobTriplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobTriplets);
    //drawBlobQuadruplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobQuadriplets);
    // drawPointsAndIds(output, mDetectionInfo.matches);
    
     if(deviceOrientation)
         drawAxes(*output, *deviceOrientation);
    
    // Draw landmark detections
    std::vector<cv::Point2f> corners(4);
    
    auto lmDetectionsIt = mDetectionInfo.landmarkDetections.cbegin();
    auto landmarksIt = mLandmarks.cbegin();
    auto colorIt = colorPalette.cbegin();
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt, ++colorIt)
    {
        const Landmark& landmark = *landmarksIt;
        const cv::Mat& h = lmDetectionsIt->getHomography();
        
        // Reset the color iterator if needed
        if(colorIt == colorPalette.cend())
            colorIt = colorPalette.cbegin();
        
        if(h.empty())
            continue;
        
        cv::perspectiveTransform(landmark.getCorners(), corners, h);
        cv::line(*output, corners[0], corners[1], *colorIt, 2);
        cv::line(*output, corners[1], corners[2], *colorIt, 2);
        cv::line(*output, corners[2], corners[3], *colorIt, 2);
        cv::line(*output, corners[3], corners[0], *colorIt, 2);
        
        for(auto c : lmDetectionsIt->getCorrespondences())
        {
            cv::Point2f p = c.second;
            cv::circle(*output, p, 2, cv::Scalar(0, 255, 255));
        }

        //draw pose
        //draw object frame (axis XYZ)
        std::vector<cv::Point3f> framePoints;
        framePoints.push_back(cv::Point3f(0,0,0));
        framePoints.push_back(cv::Point3f(0.03,0,0));
        framePoints.push_back(cv::Point3f(0,0.03,0));
        framePoints.push_back(cv::Point3f(0,0,0.03));

        //cv::Affine3d pose = lmDetectionsIt->getPose().inv();
        cv::Affine3d pose = lmDetectionsIt->getPose();
        std::vector<cv::Point2f> vprojVertices;
        cv::projectPoints(framePoints, pose.rvec(), pose.translation(), mCalibration.cameraMatrix, mCalibration.distCoeffs, vprojVertices);
        cv::line(*output, vprojVertices[0], vprojVertices[1], cv::Scalar(0,0,255), 2);
        cv::line(*output, vprojVertices[0], vprojVertices[2], cv::Scalar(0,255,0), 2);
        cv::line(*output, vprojVertices[0], vprojVertices[3], cv::Scalar(255,0,0), 2);

    }
}

}

/*
namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

//function to be called by java wrapper on tablet to store iamge sequence and corresponding orientation
void ThymioTracker::update(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{    
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());

    static int counter = 0;

    std::string path = "/sdcard/ThymioTracker/record/";
    std::string name1 = path+"img_" + patch::to_string(counter) + ".png";
    cv::imwrite(name1, input);

    std::string name2 = path+"ori_" + patch::to_string(counter) + ".dat";
    //cv::FileStorage file(name2, cv::FileStorage::WRITE);
    //file << *deviceOrientation;
    cv::FileStorage fs(name2, cv::FileStorage::WRITE);
    cv::write(fs, "deviceOrientation", *deviceOrientation);

    std::cout<<"save frame "<<counter<<std::endl;
    counter++;

}*/