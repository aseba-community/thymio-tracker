//matching by Geometric hashing

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//opencv display
//Mat inputImage;
const char window_name[] = "InputWindow";
const char window_name_patch[] = "Patch";
const char window_name_res[] = "Result";
const char window_name_searchin[] = "SearchIn";

//3D model, first simple version with just top
#include "VideoSource.hpp"
#include "GH.hpp"
#include "GHscale.hpp"
#include "Visualization3D.hpp"
#include "Grouping.hpp"

namespace tt = thymio_tracker;

void drawPointsAndIds(Mat &inputImage,vector<tt::DetectionGH> &_matches)
{
    //draw Id
    for(int i=0;i<_matches.size();i++)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", _matches[i].id);
        circle(inputImage, _matches[i].position, 4, cvScalar(0,250,250), -1, 8, 0);
        putText(inputImage, pointIdStr, _matches[i].position,FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}



//test for NCC heuristic search
//objective is to replace brisk
//so we have a set of keypoints (brisk keypoints on template)
//we have the homography at the previous frame, and from that 
//we need to find the position of the keypoints in the current image and update the homography
/*int main( int argc, char** argv )
{
    //load camera and intrinsic parameters
    VideoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,100);
    //VideoSourceLive videoSource(EmbeddedCam);
    
    //resize input
    //videoSource.resizeSource(0.33);
    videoSource.resizeSource(0.5);
    //videoSource.grabNewFrame();
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    namedWindow( window_name_patch, WINDOW_AUTOSIZE );
    namedWindow( window_name_res, WINDOW_AUTOSIZE );
    namedWindow( window_name_searchin, WINDOW_AUTOSIZE );
    
    //create feature extractor
    cv::Ptr<cv::Feature2D> mFeatureExtractor = cv::BRISK::create();

    //define out template which will be a crop of our first image
    cv::Mat templateImg;
    std::vector<cv::KeyPoint> detectedKeypoints;
    cv::Mat detectedDescriptors;
    std::vector<cv::Point2f>  templatePoints;

    //current homography
    cv::Mat mHomography;

    //which part of first image we uyse to define template
    Point2i cornerTopLeft(600,250);
    Size templateSize(200,200);

    //define size of patch around keypoints for NCC search
    //int patch_size = 9;
    //int window_size = 16;
    int patch_size = 64;
    int window_size = 32;
    int half_window_size = window_size/2;

    
    //process sequence
    Affine3d robotPose;
    int count =0;
    while(1)
    {
        //get new image
        //if(count == 0)
            videoSource.grabNewFrame();
        //Mat inputImage = videoSource.getFramePointer();
        Mat inputImage = videoSource.getFramePointer().clone();

        //convert to grayscale
        cv::Mat gray_input;
        cv::cvtColor(inputImage, gray_input, CV_RGB2GRAY);

        if(count ==0)
        {
            //extract template from first image
            cv::Rect myROI(cornerTopLeft.x, cornerTopLeft.y, templateSize.width, templateSize.height);
            templateImg = gray_input(myROI).clone();

            //extract features
            mFeatureExtractor->detectAndCompute(templateImg, cv::noArray(),
                                                detectedKeypoints, detectedDescriptors);
            //get points from them
            std::vector<cv::Point2f>  scenePoints;
            for(int i=0;i<detectedKeypoints.size();i++)
            {
                templatePoints.push_back(detectedKeypoints[i].pt);
                scenePoints.push_back(detectedKeypoints[i].pt + Point2f(cornerTopLeft));
            }

            //init first homography
            mHomography = cv::findHomography(templatePoints, scenePoints, 0);

        }

        //lets track
        //now we have the homography estimated in the previous image
        //and a set of keypoints in the template => project the keypoints in the current image
        //project the template in the current image to fill patches around keypoints
        //and do NCC search for all keypoints

        //warp points 
        std::vector<cv::Point2f>  scenePoints_tm1;
        perspectiveTransform(templatePoints,scenePoints_tm1,mHomography);

        //warp points around keypoints to estimate affine transformation of each patch
        std::vector<cv::Point2f>  templatePoints_xp1;
        for(int i=0;i<templatePoints.size();i++)
            templatePoints_xp1.push_back(Point2f(templatePoints[i].x+1.,templatePoints[i].y));
        std::vector<cv::Point2f>  scenePoints_xp1; 
        perspectiveTransform(templatePoints_xp1,scenePoints_xp1,mHomography);
        
        std::vector<cv::Point2f>  templatePoints_yp1;
        for(int i=0;i<templatePoints.size();i++)
            templatePoints_yp1.push_back(Point2f(templatePoints[i].x,templatePoints[i].y+1.));
        std::vector<cv::Point2f>  scenePoints_yp1; 
        perspectiveTransform(templatePoints_yp1,scenePoints_yp1,mHomography);


        //now for all keypoints fill patch using homography and serach for it in current image
        std::vector<cv::Point2f>  scenePoints_t;
        Point2f center_patch((patch_size-1)/2,(patch_size-1)/2);

        //allocate patch and result matrix
        Mat patchCurr = Mat::zeros( patch_size, patch_size, templateImg.type() );
        Mat resultNCC = Mat::zeros( window_size, window_size, CV_32FC1 );
        cv::Rect myROILast;

        for(int i=0;i<scenePoints_tm1.size();i++)
        {
            //compute the homography which would fill the patch of the keypoint
            //for now homography goes from template keypoint to keypoint in scene at tm1
            //instead want now homography to bring current keypoint center in scene 
            //to position patch_size/2,patch_size ie center of patch

            //get the affine transform from the current image to the patch
            //get the 3 points from current image:
            std::vector<cv::Point2f>  templatePointsAffine;
            templatePointsAffine.push_back(templatePoints[i]);
            templatePointsAffine.push_back(templatePoints_xp1[i]);
            templatePointsAffine.push_back(templatePoints_yp1[i]);

            //get the 3 points in the patch ie points in current image transalted by position center - center_patch
            Point2f translationToPatch=(center_patch-scenePoints_tm1[i]);
            std::vector<cv::Point2f>  patchPointsAffine;
            //patchPointsAffine.push_back(center_patch);
            //patchPointsAffine.push_back(scenePoints_xp1[i]+translationToPatch);
            //patchPointsAffine.push_back(scenePoints_yp1[i]+translationToPatch);
            patchPointsAffine.push_back(scenePoints_tm1[i]);
            patchPointsAffine.push_back(scenePoints_xp1[i]);
            patchPointsAffine.push_back(scenePoints_yp1[i]);

            //get the affine transform
            //Mat mAffine = cv::getAffineTransform(templatePointsAffine,patchPointsAffine);
            Mat mAffine = cv::getAffineTransform(templatePointsAffine,patchPointsAffine);
            mAffine.at<double>(0,2) += translationToPatch.x;
            mAffine.at<double>(1,2) += translationToPatch.y;


            //fill patch using template info
            warpAffine( templateImg, patchCurr, mAffine, patchCurr.size() );

            //search for it in current image
            cv::Rect myROI(scenePoints_tm1[i].x-center_patch.x-half_window_size,scenePoints_tm1[i].y-center_patch.y-half_window_size,
                            window_size+patch_size-1,window_size+patch_size-1);//region of interest is around current position of point
            //matchTemplate( gray_input(myROI), patchCurr, resultNCC, CV_TM_CCORR );
            matchTemplate( gray_input(myROI), patchCurr, resultNCC, CV_TM_CCORR_NORMED );
            normalize( resultNCC, resultNCC, 0, 1, NORM_MINMAX, -1, Mat() );

            /// Localizing the best match with minMaxLoc searching for max NCC
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            minMaxLoc( resultNCC, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

            //have location in window=> get corresponding position in image
            Point2f estPos =  Point2f(maxLoc)+scenePoints_tm1[i]+Point2f(-half_window_size,-half_window_size);
            //Point2f estPos =  Point2f(minLoc)+scenePoints_tm1[i]+Point2f(-half_window_size,-half_window_size);
            scenePoints_t.push_back(estPos);

            //for debug
            myROILast = myROI;

        }


        mHomography = cv::findHomography(templatePoints, scenePoints_t, CV_RANSAC, 30.);

        //Mat imgWarpedTemplate;
        //warpPerspective(templateImg,imgWarpedTemplate,mHomography,);


        for(int i=0;i<scenePoints_t.size();i++)
        //for(int i=scenePoints_t.size()-1;i<scenePoints_t.size();i++)
            circle(inputImage, scenePoints_t[i], 2, cvScalar(0,255,255), 1);

        for(int i=0;i<scenePoints_t.size();i++)
        //for(int i=scenePoints_t.size()-1;i<scenePoints_t.size();i++)
            line(inputImage, scenePoints_tm1[i], scenePoints_t[i], cvScalar(0,255,0), 1);
        
 
        //for(int i=0;i<detectedKeypoints.size();i++)
        //    circle(inputImage, detectedKeypoints[i].pt, 2, cvScalar(0,255,255), 1);
        
        
        //process(inputImage);
        imshow(window_name,inputImage);
        imshow(window_name_patch,patchCurr);
        imshow(window_name_res,resultNCC);
        imshow(window_name_searchin,gray_input(myROILast));


        
        //press escape to leave loop
        if(waitKey() == 27)break;
        //waitKey();
        count ++;
    }
    
    return 0;
    
}*/


int main( int argc, char** argv )
{
    //load camera and intrinsic parameters
    //VideoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,100);
    VideoSourceLive videoSource(EmbeddedCam);
    
    //resize input
    //videoSource.resizeSource(0.33);
    videoSource.resizeSource(0.5);
    //videoSource.grabNewFrame();
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //craete grouping object
    tt::Grouping mGrouping;

    
    //process sequence
    Affine3d robotPose;
    bool found=false;
    while(1)
    {
        //get new image
        videoSource.grabNewFrame();
        //Mat inputImage = videoSource.getFramePointer();
        Mat inputImage = videoSource.getFramePointer().clone();

        
        //just for visualization (its done again in PnP)
        //get the pairs which are likely to belong to group of blobs from model
        vector<cv::KeyPoint> blobs;
        vector<tt::BlobPair> blobPairs;
        mGrouping.getBlobs(inputImage,blobs);
        
        mGrouping.getPairsFromBlobs(blobs,blobPairs);
        
        //get triplet by checking homography and inertia
        vector<tt::BlobTriplet> blobTriplets;
        mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
        
        vector<tt::BlobQuadruplets> blobQuadriplets;
        mGrouping.getQuadripletsFromTriplets(blobTriplets,blobQuadriplets,true);
        
        
        //get robot s pose from image
        tt::ThymioBlobModel mRobot;
        found = mRobot.getPoseFromBlobs(blobs,videoSource.mCalibration,robotPose,!found);
        
        if(found)//draw model from found pose
            mRobot.draw(inputImage,videoSource.mCalibration, robotPose);
        else
            putText(inputImage, "Lost", Point2i(10,10),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
        
        
        //flush
        drawBlobPairs(inputImage,blobs,blobPairs);
        drawBlobTriplets(inputImage,blobs,blobTriplets);
        drawBlobQuadruplets(inputImage,blobs,blobQuadriplets);
        
        
        //process(inputImage);
        imshow(window_name,inputImage);
        
        //press escape to leave loop
        if(waitKey(5) == 27)break;
        //waitKey();
    }
    
    return 0;
    
}


