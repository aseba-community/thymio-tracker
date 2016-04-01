
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

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

void ThymioTracker::updateOrientation(const cv::Mat& input,
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

}
    
void ThymioTracker::update(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{    
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());

    //to do some debugging and plot stuff on current image
    cv::Mat cpImg;
    input.copyTo(cpImg);
    
    //mDetectionInfo.clear();
    //if robot was not found in previous image then run Geometric Hashing
    if(!mDetectionInfo.robotFound)
    {
        std::cout<<"Detection"<<std::endl;
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

        //if robot has been found init tracks
        if(mDetectionInfo.robotFound)
        {
            //for now lets just estimate the homography between top view and image
            //and do PnP from that.

            //have blob pos listed in mRobot
            //project model 3D points to get current keypoint position
            std::vector<cv::Point2f> vprojVertices;
            projectPoints(mRobot.mVertices, mDetectionInfo.robotPose.rvec(), mDetectionInfo.robotPose.translation(), mCalibration.cameraMatrix, mCalibration.distCoeffs, vprojVertices);
    
            //add to tracking correspondences
            mDetectionInfo.mRobotCorrespondences.clear();
            for(int i=0;i<vprojVertices.size();i++)
                mDetectionInfo.mRobotCorrespondences[i]=vprojVertices[i];

            //find homography
            mDetectionInfo.mRobotHomography = cv::findHomography(mRobot.mRobotKeypointPos, vprojVertices);

        }
    }
    else
    {
        std::cout<<"Track"<<std::endl;
        //robot was found in previous image => can do tracking
        //let s do active search mixed with KLT on blobs for now

        // Get positions of keypoints in previous frame
        std::vector<cv::Point2f> prevPoints;
        for(auto p : mDetectionInfo.mRobotCorrespondences)
            prevPoints.push_back(p.second);
        
        // Optical flow
        std::vector<cv::Mat> pyramid1, pyramid2;
        int maxLevel = 3;
        const cv::Size winSize = cv::Size(31, 31);
        std::vector<cv::Point2f> nextPoints;
        std::vector<unsigned char> status;
        cv::calcOpticalFlowPyrLK(mDetectionInfo.prevImage, input, prevPoints, nextPoints, status,
                                cv::noArray(), winSize, maxLevel,
                                cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.1),
                                0,
                                0.001);

        // Keep only found keypoints in mRobotCorrespondences
        std::vector<cv::Point2f> scenePoints;
        std::vector<int> correspondences;

        auto statusIt = status.cbegin();
        auto nextPointsIt = nextPoints.cbegin();
        auto prevPointsIt = prevPoints.cbegin();
        int cpt = 0;
        for(; statusIt != status.cend(); ++statusIt, ++nextPointsIt, ++prevPointsIt)
        {
            if(*statusIt)
            {
                scenePoints.push_back(*nextPointsIt);
                correspondences.push_back(cpt);

                //plot
                line(input, *prevPointsIt, *nextPointsIt, cv::Scalar(0,255,255), 1);
    
            }
            cpt++;
        }

        //search for correspondences using active search


        //find homography and pose from matches
        std::vector<cv::Point2f> objectPoints;
        for(int c : correspondences)
            objectPoints.push_back(mRobot.mRobotKeypointPos[c]);

        int minCorresp = 10;//minimum number of matches
        float ransacThreshold = 10.;
        cv::Mat homography;
        std::vector<unsigned char> mask;
        if(scenePoints.size()>minCorresp)
            homography = cv::findHomography(objectPoints, scenePoints, CV_RANSAC, ransacThreshold, mask);

        // Save homography and inliers
        mDetectionInfo.mRobotHomography = homography;
        
        mDetectionInfo.mRobotCorrespondences.clear();
        auto maskIt = mask.cbegin();
        auto correspIt = correspondences.cbegin();
        auto scenePointsIt = scenePoints.cbegin();
        for(; maskIt != mask.cend(); ++maskIt, ++correspIt, ++scenePointsIt)
        {
            if(!*maskIt)
                continue;
            
            mDetectionInfo.mRobotCorrespondences[*correspIt] = *scenePointsIt;
        }
        int nbInliers = mDetectionInfo.mRobotCorrespondences.size();

        //find corresponding pose
        if(nbInliers>minCorresp)
        {
            //transform 4 points from image of top of robot using homography
            std::vector<int> pointsForPose;
            pointsForPose.push_back(0);
            pointsForPose.push_back(3);
            pointsForPose.push_back(10);
            pointsForPose.push_back(13);

            std::vector<cv::Point2f> mCorners;
            for(int i=0;i<4;i++)mCorners.push_back(mRobot.mRobotKeypointPos[pointsForPose[i]]);
           

            std::vector<cv::Point2f> mCornersInScene;
            cv::perspectiveTransform(mCorners, mCornersInScene, homography);

            //perform PnP with corresp 3D positions
            std::vector<cv::Point3f> mModelPoints;
            for(int i=0;i<4;i++)mModelPoints.push_back(mRobot.mVertices[pointsForPose[i]]);
           
            //perform pnp
            cv::Vec3d rot_v;
            cv::Vec3d trans_v;
            cv::solvePnP(mModelPoints,mCornersInScene, mCalibration.cameraMatrix, mCalibration.distCoeffs,rot_v,trans_v);
            mDetectionInfo.robotPose = cv::Affine3d(rot_v,trans_v);
        }
        std::cout<<"Nb feat tracked = "<<nbInliers<<std::endl;

        mDetectionInfo.robotFound = (nbInliers>minCorresp);

    }
    
    static int counter = 100;   
    ++counter;
    
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


    // Extract features only once every 20 frames and only if need to do any detection
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
    
    //calibration of camera: if needed, capture a set of frames where tags are tracked and use them do calibrate the camera
    //add the matches to the calibration information
    /*static int counter_calib = 100;
    int nb_detections_for_calibration = 30;
    ++counter_calib;

    if(mCalibrationInfo.objectPoints.size() < nb_detections_for_calibration && counter_calib >= 20)
    {
        //for each tracked landmark add the matches to the calibration tool
        landmarksIt = mLandmarks.cbegin();
        lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();

        for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
        {
            const cv::Mat& h = lmDetectionsIt->getHomography();
            if(!h.empty())
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
            }
        }

        if(mCalibrationInfo.objectPoints.size() >= nb_detections_for_calibration)
        {
            std::vector<cv::Mat> rotationVectors;
            std::vector<cv::Mat> translationVectors;

            cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

            int flags = 0;
            double rms = calibrateCamera(mCalibrationInfo.objectPoints, mCalibrationInfo.imagePoints, input.size(), mCalibration.cameraMatrix,
                          mCalibration.distCoeffs, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

            std::cout<<"camera calibration RMS = "<<rms<<std::endl;
        }

        counter_calib = 0;
    }*/

    cpImg.copyTo(mDetectionInfo.prevImage);
    //input.copyTo(mDetectionInfo.prevImage);
    
    mTimer.tic();
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
    
    if(mDetectionInfo.robotFound)
        mRobot.draw(*output, mCalibration, mDetectionInfo.robotPose);
    else
        putText(*output, "Lost",
                cv::Point2i(10,10),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0,0,250), 1, CV_AA);
    
    // drawBlobPairs(*output, mDetectionInfo.blobs, mDetectionInfo.blobPairs);
    // drawBlobTriplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobTriplets);
    // drawBlobQuadruplets(*output, mDetectionInfo.blobs, mDetectionInfo.blobQuadriplets);
    // // drawPointsAndIds(output, mDetectionInfo.matches);
    
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
