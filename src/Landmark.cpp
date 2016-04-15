
#include "Landmark.hpp"

#include <stdexcept>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <iostream>

namespace thymio_tracker
{

Landmark Landmark::fromFileStorage(cv::FileStorage& fs)
{
    cv::Mat image;
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // cv::Size2f realSize;
    std::vector<float> realSize;
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    
    cv::read(fs["keypoints"], keypoints);
    cv::read(fs["descriptors"], descriptors);
    // cv::read(fs["image_size"], imageSize);
    cv::read(fs["real_size"], realSize);
    cv::read(fs["image"], image);

    
    if(image.empty())
        throw std::runtime_error("Could not load image data");
    
    return Landmark(image, keypoints, descriptors, cv::Size(realSize[0], realSize[1]));
}

Landmark::Landmark(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const cv::Mat& descriptors,
                    const cv::Size2f& realSize)
    : mImage(image)
    , mKeypoints(keypoints)
    , mKeypointPos(keypoints.size())
    , mDescriptors(descriptors)
    , mRealSize(realSize)
    , mMatcher(cv::NORM_HAMMING)
{
    cv::buildOpticalFlowPyramid(image, mPyramid, cv::Size(21, 21), 0);
    
    std::transform(mKeypoints.begin(), mKeypoints.end(), mKeypointPos.begin(),
        [](const cv::KeyPoint& kp){return kp.pt;});
}

void Landmark::find(const cv::Mat& image,
              const cv::Mat& prevImage,
              const IntrinsicCalibration& mCalibration,
              const std::vector<cv::KeyPoint>& keypoints,
              const cv::Mat& descriptors,
              LandmarkDetection& detection) const
{
    std::vector<cv::Point2f> scenePoints;
    std::vector<int> correspondences;
    
    if(detection.mCorrespondences.empty())
    {
        if(!descriptors.empty())
            this->findCorrespondencesWithKeypoints(keypoints, descriptors, scenePoints, correspondences);
    }
    else
    {
        this->findCorrespondencesWithTracking(image, prevImage, detection, scenePoints, correspondences);
        this->findCorrespondencesWithActiveSearch(image, detection , scenePoints, correspondences);
    }
    
    std::vector<cv::Point2f> objectPoints;
    for(int c : correspondences)
        objectPoints.push_back(mKeypoints[c].pt);
    
    // Compute homography
    unsigned int minCorresp = 20;//minimum number of matches
    float ransacThreshold = 5.;
    cv::Mat homography;
    std::vector<unsigned char> mask;
    //if(!scenePoints.empty())
    if(scenePoints.size()>minCorresp)
        homography = cv::findHomography(objectPoints, scenePoints, CV_RANSAC, ransacThreshold, mask);


    //need to recompute outliers as the ones from above are those from the ransac estimation without refinement
    std::vector<cv::Point2f> scenePointsRefined;
    //if(!scenePoints.empty())
    if(scenePoints.size()>minCorresp)
    {
        //matches for refining the homography with inliers from second round (to decrease jitter)
        std::vector<cv::Point2f> objectPointsValid;
        std::vector<cv::Point2f> scenePointsValid;

        cv::perspectiveTransform(objectPoints, scenePointsRefined, homography);
        for(unsigned int i=0;i<scenePointsRefined.size();i++)
        {
            mask[i]=(unsigned char)(cv::norm(scenePointsRefined[i]-scenePoints[i])<ransacThreshold);
            if(mask[i])
            {
                objectPointsValid.push_back(objectPoints[i]);
                scenePointsValid.push_back(scenePoints[i]);
            }
        }

        //second round refinement
        homography = cv::findHomography(objectPointsValid, scenePointsValid);
    }
    
    // Save homography and inliers
    detection.mHomography = homography;
    
    detection.mCorrespondences.clear();
    auto maskIt = mask.cbegin();
    auto correspIt = correspondences.cbegin();
    auto scenePointsIt = scenePoints.cbegin();
    for(; maskIt != mask.cend(); ++maskIt, ++correspIt, ++scenePointsIt)
    {
        if(!*maskIt)
            continue;
        
        detection.mCorrespondences[*correspIt] = *scenePointsIt;
    }

    //pose computation
    if(scenePoints.size()>minCorresp)
    {
        //use standard Pnp: define model
        std::vector<cv::Point3f> mModelPoints(4);
        //divide everything by 2x100 to get half size in meter
        mModelPoints[3] = cv::Point3f(-mRealSize.width/200., -mRealSize.height/200.,0.);
        mModelPoints[2] = cv::Point3f(mRealSize.width/200., -mRealSize.height/200.,0.);
        mModelPoints[1] = cv::Point3f(mRealSize.width/200., mRealSize.height/200.,0.);
        mModelPoints[0] = cv::Point3f(-mRealSize.width/200., mRealSize.height/200.,0.);

        //project corners in image
        std::vector<cv::Point2f> mCornersInScene;
        cv::perspectiveTransform(getCorners(), mCornersInScene, homography);

        //perform pnp
        cv::Vec3d rot_v;
        cv::Vec3d trans_v;
        cv::solvePnP(mModelPoints,mCornersInScene, mCalibration.cameraMatrix, mCalibration.distCoeffs,rot_v,trans_v);
        detection.mPose = cv::Affine3d(rot_v,trans_v);

    }

}

void Landmark::findCorrespondencesWithKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                const cv::Mat& descriptors,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    std::vector<std::vector<cv::DMatch> > matches;
    
    mMatcher.knnMatch(descriptors, mDescriptors, matches, 2);
    
    // Keep only significant matches
    std::vector<cv::DMatch> goodMatches;
    for(auto el : matches)
    {
        if(el[0].distance < 0.75 * el[1].distance)
            goodMatches.push_back(el[0]);
    }
    
    for(auto match : goodMatches)
    {
        correspondences.push_back(match.trainIdx);
        scenePoints.push_back(keypoints[match.queryIdx].pt);
    }
}

void Landmark::findCorrespondencesWithActiveSearch(const cv::Mat& image,
                                const LandmarkDetection& prevDetection,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    //project all the keypoints using previous homography, 
    //fill patches 9x9 patches using template warped over current image
    //search for the patches in current image and 16x16 window
    int patch_size = 9;
    int window_size = 16;

    int half_window_size = window_size/2;
    int half_patch_size = (patch_size-1)/2;

    //will pick a random subset of the keypoint features to do active search
    //so that after a few frames, all features will be covered and shift free
    std::vector<int> myIndexes;
    for (unsigned int i=0; i<mKeypointPos.size(); i++) myIndexes.push_back(i);
    std::random_shuffle ( myIndexes.begin(), myIndexes.end() );

    int nbKeypointsCoveredPerFrame = 50;
    for(int i = 0; i < nbKeypointsCoveredPerFrame && i < myIndexes.size(); i++)
    {
        int kpIndex = myIndexes[i];
        cv::Point2f p = mKeypointPos[kpIndex];
        //for each keypoint in template, define reference points to warp using homography
        //and get corresponding affine transformation
        std::vector<cv::Point2f> templateFramePoints;
        templateFramePoints.push_back(p);
        templateFramePoints.push_back(cv::Point2f(p.x+1.,p.y));
        templateFramePoints.push_back(cv::Point2f(p.x,p.y+1.));

        //transform this frame using previous homography
        std::vector<cv::Point2f> sceneFramePoints;
        cv::perspectiveTransform(templateFramePoints,sceneFramePoints,prevDetection.mHomography);

        //find affine transformation from template to scene frame
        cv::Mat mAffine = cv::getAffineTransform(templateFramePoints,sceneFramePoints);
        //need to map to patch not current frame
        mAffine.at<double>(0,2) += half_patch_size - sceneFramePoints[0].x;
        mAffine.at<double>(1,2) += half_patch_size - sceneFramePoints[0].y;

        //fill patch using template info
        cv::Mat patchCurr = cv::Mat::zeros( patch_size, patch_size, mImage.type() );
        cv::warpAffine( mImage, patchCurr, mAffine, patchCurr.size() );

        //search for it in current image
        //cv::Rect myROI(sceneFramePoints[0].x-half_patch_size-half_window_size,sceneFramePoints[0].y-half_patch_size-half_window_size,
        //                window_size+patch_size-1,window_size+patch_size-1);//region of interest is around current position of point
        //cv::Mat resultNCC = cv::Mat::zeros( window_size, window_size, CV_32FC1 );
        //Problem when point is on border with previous code
        int myRoi_l = sceneFramePoints[0].x-half_patch_size-half_window_size; myRoi_l = (myRoi_l<0)?0:myRoi_l;
        int myRoi_t = sceneFramePoints[0].y-half_patch_size-half_window_size; myRoi_t = (myRoi_t<0)?0:myRoi_t;
        int myRoi_r = sceneFramePoints[0].x+half_patch_size+half_window_size; myRoi_r = (myRoi_r>image.size().width)?image.size().width:myRoi_r;
        int myRoi_d = sceneFramePoints[0].y+half_patch_size+half_window_size; myRoi_d = (myRoi_d>image.size().height)?image.size().height:myRoi_d;       

        cv::Rect myROI(myRoi_l,myRoi_t,myRoi_r-myRoi_l,myRoi_d-myRoi_t);//region of interest is around current position of point

        //verify that the search region is valid
        int result_cols = myROI.size().height - patch_size + 1;
        int result_rows = myROI.size().width - patch_size + 1;

        if(result_cols > half_window_size && result_rows > half_window_size)
        {
            cv::Mat resultNCC = cv::Mat::zeros( result_cols, result_rows, CV_32FC1 );

            cv::matchTemplate( image(myROI), patchCurr, resultNCC, CV_TM_CCORR_NORMED );
            //cv::normalize( resultNCC, resultNCC, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

            /// Localizing the best match with minMaxLoc searching for max NCC
            double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
            cv::minMaxLoc( resultNCC, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
            //std::cout<<maxVal<<std::endl;

            if(maxVal>0.9)
            {
                //have location in window=> get corresponding position in image
                //cv::Point2f estPos =  cv::Point2f(maxLoc)+sceneFramePoints[0]+cv::Point2f(-half_window_size,-half_window_size);
                cv::Point2f estPos =  cv::Point2f(maxLoc)+cv::Point2f(myRoi_l+half_patch_size,myRoi_t+half_patch_size);
                
                //add it to list of matches
                correspondences.push_back(kpIndex);
                scenePoints.push_back(estPos);
            }
        }

    }

}

void Landmark::findCorrespondencesWithTracking(const cv::Mat& image,
                                const cv::Mat& prevImage,
                                const LandmarkDetection& prevDetection,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    if(prevImage.empty() || prevDetection.mCorrespondences.empty())
        return;
    
    // Get positions of keypoints in previous frame
    std::vector<cv::Point2f> prevPoints;
    for(auto p : prevDetection.mCorrespondences)
        prevPoints.push_back(p.second);
    
    // Optical flow
    int maxLevel = 3;
    const cv::Size winSize = cv::Size(21, 21);
    std::vector<cv::Point2f> nextPoints;
    std::vector<unsigned char> status;
    cv::calcOpticalFlowPyrLK(prevImage, image, prevPoints, nextPoints, status,
                            cv::noArray(), winSize, maxLevel,
                            cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.1),
                            0,
                            0.001);
    
    // Keep only found keypoints
    /*auto statusIt = status.cbegin();
    auto nextPointsIt = nextPoints.cbegin();
    auto correspIt = prevDetection.mCorrespondences.cbegin();
    for(; statusIt != status.cend(); ++statusIt, ++nextPointsIt, ++correspIt)
    {
        if(!*statusIt)
            continue;
        
        scenePoints.push_back(*nextPointsIt);
        correspondences.push_back(correspIt->first);
    }*/

    //pick up random subset of tracked features and do sanity check based on NCC
    auto statusIt = status.cbegin();
    auto nextPointsIt = nextPoints.cbegin();
    auto correspIt = prevDetection.mCorrespondences.cbegin();

    //random selection of points
    int nbTracksChecked = 50;
    float NCCvalid = 0.9;
    std::vector<int> myIndexes;
    for (unsigned int i=0; i<nextPoints.size(); i++) myIndexes.push_back(i);
    std::random_shuffle ( myIndexes.begin(), myIndexes.end() );
    auto indexIt = myIndexes.cbegin();

    int patch_size = 9;
    int half_patch_size = (patch_size-1)/2;
    for(; statusIt != status.cend(); ++statusIt, ++nextPointsIt, ++correspIt, ++indexIt)
    {
        if(!*statusIt)
            continue;

        if(*indexIt>nbTracksChecked)
        {        
            //portion of tarcks that we don't check => just add them
            scenePoints.push_back(*nextPointsIt);
            correspondences.push_back(correspIt->first);
        }
        else
        {
            int kpIndex = correspIt->first;
            cv::Point2f p = mKeypointPos[kpIndex];
            //for each keypoint in template, define reference points to warp using homography
            //and get corresponding affine transformation
            std::vector<cv::Point2f> templateFramePoints;
            templateFramePoints.push_back(p);
            templateFramePoints.push_back(cv::Point2f(p.x+1.,p.y));
            templateFramePoints.push_back(cv::Point2f(p.x,p.y+1.));

            //transform this frame using previous homography
            std::vector<cv::Point2f> sceneFramePoints;
            cv::perspectiveTransform(templateFramePoints,sceneFramePoints,prevDetection.mHomography);

            //find affine transformation from template to scene frame
            cv::Mat mAffine = cv::getAffineTransform(templateFramePoints,sceneFramePoints);
            //need to map to patch not current frame
            mAffine.at<double>(0,2) += half_patch_size - sceneFramePoints[0].x;
            mAffine.at<double>(1,2) += half_patch_size - sceneFramePoints[0].y;

            //fill patch using template info
            cv::Mat patchCurr = cv::Mat::zeros( patch_size, patch_size, mImage.type() );
            cv::warpAffine( mImage, patchCurr, mAffine, patchCurr.size() );

            //and now just need to compare it with patch from current image centered on track
            int myRoi_l = nextPointsIt->x-half_patch_size; myRoi_l = (myRoi_l<0)?0:myRoi_l;
            int myRoi_t = nextPointsIt->y-half_patch_size; myRoi_t = (myRoi_t<0)?0:myRoi_t;
            int myRoi_r = nextPointsIt->x+half_patch_size+1; myRoi_r = (myRoi_r>image.size().width)?image.size().width:myRoi_r;
            int myRoi_d = nextPointsIt->y+half_patch_size+1; myRoi_d = (myRoi_d>image.size().height)?image.size().height:myRoi_d;       

            cv::Rect myROI(myRoi_l,myRoi_t,myRoi_r-myRoi_l,myRoi_d-myRoi_t);//region of interest is around current position of point

            //verify that the search region is valid
            int result_cols = myROI.size().height - patch_size + 1;
            int result_rows = myROI.size().width - patch_size + 1;

            //if track not on border of image then result should be 1x1
            if(result_cols > 0 && result_rows > 0)
            {
                cv::Mat resultNCC = cv::Mat::zeros( result_cols, result_rows, CV_32FC1 );
                cv::matchTemplate( image(myROI), patchCurr, resultNCC, CV_TM_CCORR_NORMED );
                if(resultNCC.at<float>(0,0)>NCCvalid)
                {
                    scenePoints.push_back(*nextPointsIt);
                    correspondences.push_back(correspIt->first);                    
                }

            }
        }
    }   
}

cv::Mat Landmark::findHomography(const std::vector<cv::KeyPoint>& keypoints,
                                    const cv::Mat& descriptors) const
{
    std::vector<std::vector<cv::DMatch> > matches;
    
    mMatcher.knnMatch(descriptors, mDescriptors, matches, 2);
    
    // Keep only significant matches
    std::vector<cv::DMatch> goodMatches;
    for(auto el : matches)
    {
        if(el[0].distance < 0.75 * el[1].distance)
            goodMatches.push_back(el[0]);
    }
    
    // Fit a homography.
    std::vector<cv::Point2f> object_points;
    std::vector<cv::Point2f> scenePoints;
    for(auto match : goodMatches)
    {
        object_points.push_back(mKeypoints[match.trainIdx].pt);
        scenePoints.push_back(keypoints[match.queryIdx].pt);
    }
    
    cv::Mat homography;
    if(!scenePoints.empty())
        homography = cv::findHomography(object_points, scenePoints, CV_RANSAC);
    
    return homography;
}

std::vector<cv::Point2f> Landmark::getCorners() const
{
    std::vector<cv::Point2f> res(4);
    
    const cv::Size size = mImage.size();
    
    res[0] = cv::Point2f(0, 0);
    res[1] = cv::Point2f(size.width, 0);
    res[2] = cv::Point2f(size.width, size.height);
    res[3] = cv::Point2f(0, size.height);
    
    return res;
}

}
