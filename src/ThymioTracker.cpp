
#include "ThymioTracker.h"

#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

void DetectionInfo::clear()
{
    blobs.clear();
    blobPairs.clear();
    blobTriplets.clear();
    blobQuadriplets.clear();
    blobsinTriplets.clear();
    matches.clear();
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
    static const cv::Scalar axes_colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    cv::Size::value_type width = image.size().width;
    cv::Point2d center(width * 0.1, width * 0.1);
    double length = width * 0.05;
    
    for(int i = 0; i < 3; ++i)
    {
        const cv::Point2d direction(orientation.at<float>(i, 1), orientation.at<float>(i, 0));
        const cv::Point2d arrow = center - length * direction;
        const cv::Scalar& color = axes_colors[i];
        
        cv::line(image, center, arrow, color);
    }
}

void loadCalibration(const std::string& filename,
                     IntrinsicCalibration* calibration)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    readCalibrationFromFileStorage(fs, *calibration);
    fs.release();
}

ThymioTracker::ThymioTracker(const std::string& calibrationFile,
                             const std::string& geomHashingFile,
                             const std::vector<std::string>& landmarkFiles)
    : mCalibrationFile(calibrationFile)
    , mGeomHashingFile(geomHashingFile)
    , mDetectionInfo(landmarkFiles.size())
    // , mFeatureExtractor(cv::ORB::create(1000))
    , mFeatureExtractor(cv::BRISK::create())
    // , mFeatureExtractor(new brisk::BriskFeature(5.0, 4))
    // , mFeatureExtractor(cv::xfeatures2d::SIFT::create())
    // , mFeatureExtractor(cv::xfeatures2d::SURF::create())
    // , mFeatureExtractor(cv::xfeatures2d::DAISY::create())
{
    mGH.loadFromFile(mGeomHashingFile);
    loadCalibration(mCalibrationFile, &mCalibration);
    mGH.setCalibration(mCalibration);
    
    // Load landmarks
    for(auto landmarkFile : landmarkFiles)
        mLandmarks.push_back(Landmark::fromFile(landmarkFile));
}

void ThymioTracker::resizeCalibration(const cv::Size& imgSize)
{
    // loadCalibration(mCalibrationFile, imgSize, &mCalibration);
    rescaleCalibration(mCalibration, imgSize);
    mGH.setCalibration(mCalibration);
}

void ThymioTracker::update(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{    
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());
    
    mDetectionInfo.clear();

    //get the pairs which are likely to belong to group of blobs from model
    mGrouping.getBlobsAndPairs(input,
                               mDetectionInfo.blobs,
                               mDetectionInfo.blobPairs);
    
    // get triplet by checking homography and inertia
    mGrouping.getTripletsFromPairs(mDetectionInfo.blobs,
                                   mDetectionInfo.blobPairs,
                                   mDetectionInfo.blobTriplets);
    
    //get only blobs found in triplets
    getBlobsInTriplets(mDetectionInfo.blobs,
                       mDetectionInfo.blobTriplets,
                       mDetectionInfo.blobsinTriplets);
    
    mGrouping.getQuadripletsFromTriplets(mDetectionInfo.blobTriplets,
                                         mDetectionInfo.blobQuadriplets);
    
    //extract blobs and identify which one fit model, return set of positions and Id
    mGH.getModelPointsFromImage(mDetectionInfo.blobsinTriplets, mDetectionInfo.matches);
    
    mDetectionInfo.robotFound = mRobot.getPose(mCalibration,
                                               mDetectionInfo.matches,
                                               mDetectionInfo.robotPose,
                                               mDetectionInfo.robotFound);
    
    static int counter = 100;
    
    ++counter;
    cv::Mat prevIm;input.copyTo(prevIm);//just to do some displaying and debugging and print stuff in input
    
    // Landmark tracking
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat detectedDescriptors;

    //check if all the landmarks are tracked
    bool allTracked = true;
    auto lmcDetectionsIt = mDetectionInfo.landmarkDetections.cbegin();
    for(; lmcDetectionsIt != mDetectionInfo.landmarkDetections.cend(); ++lmcDetectionsIt)
    {
        const cv::Mat& h = lmcDetectionsIt->getHomography();
        if(h.empty()) 
            allTracked = false;
    }


    // Extract features only once every 100 frames and only if need to do any detection
    if(!allTracked && counter >= 20)
    {
        cv::Mat gray_input;
        cv::cvtColor(input, gray_input, CV_RGB2GRAY);
        mFeatureExtractor->detectAndCompute(gray_input, cv::noArray(),
                                            detectedKeypoints, detectedDescriptors);
        counter = 0;
    }
    
    auto landmarksIt = mLandmarks.cbegin();
    auto lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
        landmarksIt->find(input, mDetectionInfo.prevImage, mCalibration, detectedKeypoints, detectedDescriptors, *lmDetectionsIt);
    
    //input.copyTo(mDetectionInfo.prevImage);
    prevIm.copyTo(mDetectionInfo.prevImage);
    
    mTimer.tic();
}

void ThymioTracker::drawLastDetection(cv::Mat* output) const
{
    // mDetectionInfo.image.copyTo(*output);
    
    if(mDetectionInfo.robotFound)
        mRobot.draw(*output, mCalibration, mDetectionInfo.robotPose);
    else
        putText(*output, "Lost",
                cv::Point2i(10,10),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0,0,250), 1, CV_AA);
    
    drawBlobPairs(*output, mDetectionInfo.blobs, mDetectionInfo.blobPairs);
    drawBlobTriplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobTriplets);
    drawBlobQuadruplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobQuadriplets);
    // drawPointsAndIds(output, mDetectionInfo.matches);
    
    // if(deviceOrientation)
    //     drawAxes(*output, *deviceOrientation);
    
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
