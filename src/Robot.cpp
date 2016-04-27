
#include "Robot.hpp"

#include <stdexcept>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <iostream>

namespace thymio_tracker
{


void Robot::init(IntrinsicCalibration *_mCalibration_ptr,
              cv::FileStorage& geomHashingStorage,
              cv::FileStorage& robotModelStorage)
{
    mCalibration_ptr = _mCalibration_ptr;
    //mGH.loadFromStream(geomHashingStream);
    mGH.loadFromFileStorage(geomHashingStorage);
    mGH.setCalibration(mCalibration_ptr);

    mModel.loadTrackingModel(robotModelStorage);
}


void Robot::find(const cv::Mat& input,
          const cv::Mat& prevImage,
          RobotDetection& mDetectionInfo) const
{
    mDetectionInfo.clearBlobs();
    IntrinsicCalibration& mCalibration = *mCalibration_ptr;

    //if robot was not found in previous image then run Geometric Hashing
    if(!mDetectionInfo.isFound())
    {
        this->findFromBlobGroupsAndGH(input,mDetectionInfo);

        //if robot has been found init tracks
        if(mDetectionInfo.robotFound)
        {
            //for now lets just estimate the homography between top view and image
            //and do PnP from that.

            //have blob pos listed in mRobot
            //project model 3D points to get current keypoint position
            std::vector<cv::Point2f> vprojVertices;
            projectPoints(mModel.mVertices, mDetectionInfo.mPose.rvec(), mDetectionInfo.mPose.translation(), mCalibration.cameraMatrix, mCalibration.distCoeffs, vprojVertices);
    
            //add to tracking correspondences
            mDetectionInfo.mCorrespondences.clear();
            for(unsigned int i=0;i<vprojVertices.size();i++)
                mDetectionInfo.mCorrespondences[i]=vprojVertices[i];

            //find homography
            mDetectionInfo.mHomography = cv::findHomography(mModel.mVerticesTopPos, vprojVertices);

        }
    }
    else
    {
        //robot was found in previous image => can do tracking
        //let s do active search mixed with KLT on blobs for now
        std::vector<cv::Point2f> scenePoints;
        std::vector<int> correspondences;

        //search with KLT
        this->findCorrespondencesWithTracking(input, prevImage, mDetectionInfo, scenePoints, correspondences);
        this->findCorrespondencesWithActiveSearch(input, mDetectionInfo , scenePoints, correspondences);


        //search for correspondences using active search
        //TODO

        //find homography and pose from matches
        std::vector<cv::Point2f> objectPoints;
        for(int c : correspondences)
            objectPoints.push_back(mModel.mRobotKeypointPos[c]);

        unsigned int minCorresp = 10;//minimum number of matches
        float ransacThreshold = 10.;
        cv::Mat homography;
        std::vector<unsigned char> mask;
        if(scenePoints.size()>minCorresp)
            homography = cv::findHomography(objectPoints, scenePoints, CV_RANSAC, ransacThreshold, mask);

        // Save homography and inliers
        mDetectionInfo.mHomography = homography;
        
        mDetectionInfo.mCorrespondences.clear();
        auto maskIt = mask.cbegin();
        auto correspIt = correspondences.cbegin();
        auto scenePointsIt = scenePoints.cbegin();
        for(; maskIt != mask.cend(); ++maskIt, ++correspIt, ++scenePointsIt)
        {
            if(!*maskIt)
                continue;
            
            mDetectionInfo.mCorrespondences[*correspIt] = *scenePointsIt;
        }
        unsigned int nbInliers = mDetectionInfo.mCorrespondences.size();

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
            for(int i=0;i<4;i++)mCorners.push_back(mModel.mRobotKeypointPos[pointsForPose[i]]);
           

            std::vector<cv::Point2f> mCornersInScene;
            cv::perspectiveTransform(mCorners, mCornersInScene, homography);

            //get the 3D coordinates of the corresponding model vertices
            std::vector<cv::Point3f> mModelPoints;
            for(int i=0;i<4;i++)mModelPoints.push_back(mModel.mVertices[pointsForPose[i]]);
           
            //perform pnp
            cv::Vec3d rot_v;
            cv::Vec3d trans_v;
            cv::solvePnP(mModelPoints,mCornersInScene, mCalibration.cameraMatrix, mCalibration.distCoeffs,rot_v,trans_v);
            mDetectionInfo.mPose = cv::Affine3d(rot_v,trans_v);
        }
        //std::cout<<"Nb feat tracked = "<<nbInliers<<std::endl;

        mDetectionInfo.robotFound = (nbInliers>minCorresp);

    }

}

void Robot::findFromBlobGroupsAndGH(const cv::Mat& image,
                                 RobotDetection& mDetectionInfo) const
{
    //get the pairs which are likely to belong to group of blobs from model
    mGrouping.getBlobsAndPairs(image,
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
    
    mDetectionInfo.robotFound = mModel.getPose(*mCalibration_ptr,
                                               mDetectionInfo.matches,
                                               mDetectionInfo.mPose,
                                               mDetectionInfo.robotFound);
}


void Robot::findCorrespondencesWithTracking(const cv::Mat& image,
                                const cv::Mat& prevImage,
                                const RobotDetection& prevDetection,
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
    const cv::Size winSize = cv::Size(31, 31);
    std::vector<cv::Point2f> nextPoints;
    std::vector<unsigned char> status;
    cv::calcOpticalFlowPyrLK(prevImage, image, prevPoints, nextPoints, status,
                            cv::noArray(), winSize, maxLevel,
                            cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.1),
                            0,
                            0.001);

    // Keep only found keypoints in mCorrespondences

    /*auto statusIt = status.cbegin();
    auto nextPointsIt = nextPoints.cbegin();
    auto correspIt = prevDetection.mCorrespondences.cbegin();

    auto prevPointsIt = prevPoints.cbegin();//for printing only
    for(; statusIt != status.cend(); ++statusIt, ++nextPointsIt, ++correspIt, ++prevPointsIt)
    {
        if(*statusIt)
        {
            scenePoints.push_back(*nextPointsIt);
            correspondences.push_back(correspIt->first);

            //plot
            //line(image, *prevPointsIt, *nextPointsIt, cv::Scalar(0,255,255), 1);

        }
    }*/

        //pick up random subset of tracked features and do sanity check based on NCC
    auto statusIt = status.cbegin();
    auto nextPointsIt = nextPoints.cbegin();
    auto correspIt = prevDetection.mCorrespondences.cbegin();

    //random selection of points
    int nbTracksChecked = 5;
    float NCCvalid = 0.9;
    std::vector<int> myIndexes;
    for (unsigned int i=0; i<nextPoints.size(); i++) myIndexes.push_back(i);
    std::random_shuffle ( myIndexes.begin(), myIndexes.end() );
    auto indexIt = myIndexes.cbegin();

    int patch_size = 29;
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
            cv::Point2f p = mModel.mRobotKeypointPos[kpIndex];
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
            cv::Mat patchCurr = cv::Mat::zeros( patch_size, patch_size, mModel.mImage.type() );
            cv::warpAffine( mModel.mImage, patchCurr, mAffine, patchCurr.size() );

            int margin = half_patch_size;
            if(sceneFramePoints[0].x < -margin || sceneFramePoints[0].y < -margin 
                || sceneFramePoints[0].x > image.size().width+margin || sceneFramePoints[0].y > image.size().height+margin )
                continue;

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

                    /*char pointIdStr[100];
                    sprintf(pointIdStr, "%f", resultNCC.at<float>(0,0));
                    circle(image, sceneFramePoints[0], 4, cvScalar(0,120,250), -1, 8, 0);
                    putText(image, pointIdStr, sceneFramePoints[0],cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(120,120,250), 1, CV_AA);
                    */
                }

            }
        }
    }  

}

void Robot::findCorrespondencesWithActiveSearch(const cv::Mat& image,
                                const RobotDetection& prevDetection,
                                std::vector<cv::Point2f>& scenePoints,
                                std::vector<int>& correspondences) const
{
    //project all the keypoints using previous homography, 
    //fill patches 9x9 patches using template warped over current image
    //search for the patches in current image and 16x16 window
    int patch_size = 29;
    int window_size = 39;

    int half_window_size = window_size/2;
    int half_patch_size = (patch_size-1)/2;

    //will pick a random subset of the keypoint features to do active search
    //so that after a few frames, all features will be covered and shift free
    std::vector<int> myIndexes;
    for (unsigned int i=0; i<mModel.mRobotKeypointPos.size(); i++) myIndexes.push_back(i);
    std::random_shuffle ( myIndexes.begin(), myIndexes.end() );

    unsigned int nbKeypointsCoveredPerFrame = 20;
    for(unsigned int i = 0; i < nbKeypointsCoveredPerFrame && i < myIndexes.size(); i++)
    {
        int kpIndex = myIndexes[i];
        cv::Point2f p = mModel.mRobotKeypointPos[kpIndex];
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
        cv::Mat patchCurr = cv::Mat::zeros( patch_size, patch_size, mModel.mImage.type() );
        cv::warpAffine( mModel.mImage, patchCurr, mAffine, patchCurr.size() );

        //search for it in current image
        //cv::Rect myROI(sceneFramePoints[0].x-half_patch_size-half_window_size,sceneFramePoints[0].y-half_patch_size-half_window_size,
        //                window_size+patch_size-1,window_size+patch_size-1);//region of interest is around current position of point
        //cv::Mat resultNCC = cv::Mat::zeros( window_size, window_size, CV_32FC1 );
        //Problem when point is on border with previous code
        int margin = half_patch_size+half_window_size;
        if(sceneFramePoints[0].x < -margin || sceneFramePoints[0].y < -margin 
            || sceneFramePoints[0].x > image.size().width+margin || sceneFramePoints[0].y > image.size().height+margin )
            continue;

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

                //plot
                //char pointIdStr[100];
                //sprintf(pointIdStr, "%f", maxVal);
                //circle(image, estPos, 4, cvScalar(0,250,250), -1, 8, 0);
                //putText(image, pointIdStr, estPos,cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,0,250), 1, CV_AA);

                //line(image, sceneFramePoints[0], estPos, cv::Scalar(0,255,125), 2);
            }
        }

    }

}
void RobotDetection::drawBlobs(cv::Mat* output) const
{
    drawBlobPairs(*output, blobs, blobPairs);
    drawBlobTriplets(*output, blobs, blobTriplets);
    drawBlobQuadruplets(*output, blobs, blobQuadriplets);
}
void RobotDetection::clearBlobs()
{
    blobs.clear();
    blobPairs.clear();
    blobTriplets.clear();
    blobQuadriplets.clear();
    blobsinTriplets.clear();
    matches.clear();
}

}