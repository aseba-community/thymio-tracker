//matching by Geometric hashing

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//opencv display
//Mat inputImage;
const char window_name[] = "InputWindow";

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
    for(unsigned int i=0;i<_matches.size();i++)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", _matches[i].id);
        circle(inputImage, _matches[i].position, 4, cvScalar(0,250,250), -1, 8, 0);
        putText(inputImage, pointIdStr, _matches[i].position,FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}

void doGHmatching()
{
    //load camera and intrinsic parameters
    //videoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    VideoSourceLive videoSource(EmbeddedCam);
    
    //resize input
    //videoSource.resizeSource(0.33);
    videoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );

#define USE_SCALE

#ifndef USE_SCALE
    tt::GH mGH(videoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GH_Arth_Perspective.dat";
#else
    tt::GHscale mGH(videoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GHscale_Arth_Perspective.dat";
#endif
    {
        //load perspective training GH
        std::ifstream geomHashingStream(GHfilename, std::ios::in | std::ios::binary);
        mGH.loadFromStream(geomHashingStream);
    }
 
    //process sequence
    Affine3d robotPose;
    bool found=false;
    while(1)
    {
        //get new image
        videoSource.grabNewFrame();
        Mat inputImage = videoSource.getFramePointer();

        clock_t startTime = clock();
        
        //extract blobs and identify which one fit model, return set of positions and Id
        vector<tt::DetectionGH> mMatches;
        mGH.getModelPointsFromImage(inputImage,mMatches);
        
        //compute robots pose
        tt::ThymioBlobModel mRobot;
        found = mRobot.getPose(videoSource.mCalibration,mMatches,robotPose,!found);
        
        cout << double( 1000.*(clock() - startTime) ) / (double)CLOCKS_PER_SEC<< " ms." << endl;
        
        if(found)//draw model from found pose
            mRobot.draw(inputImage,videoSource.mCalibration, robotPose);
        else
            putText(inputImage, "Lost", Point2i(10,10),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
        
        //flush
        drawPointsAndIds(inputImage,mMatches);
        
        //process(inputImage);
        imshow(window_name,inputImage);
        
        //press escape to leave loop
        if(waitKey(5) == 27)break;
        //waitKey();
    }
}


void searchGoodPairs()
{
    //load camera and intrinsic parameters
    //videoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    VideoSourceLive videoSource(EmbeddedCam);
    
    //resize input
    //videoSource.resizeSource(0.33);
    videoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //craete grouping object
    tt::Grouping mGrouping;
    while(1)
    {
        //get new image
        videoSource.grabNewFrame();
        Mat inputImage = videoSource.getFramePointer();
        
        //get the pairs which are likely to belong to group of blobs from model
        vector<KeyPoint> blobs;
        vector<tt::BlobPair> blobPairs;
        mGrouping.getBlobsAndPairs(inputImage,blobs,blobPairs);
        
        //get triplet by checking homography and inertia
        vector<tt::BlobTriplet> blobTriplets;
        mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
        
        vector<tt::BlobQuadruplets> blobQuadriplets;
        mGrouping.getQuadripletsFromTriplets(blobTriplets,blobQuadriplets);
        
        
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
    
}

void GoodPairsAndGH()
{
    //load camera and intrinsic parameters
    //videoSourceSeq videoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    VideoSourceLive videoSource(EmbeddedCam);
    
    //resize input
    //videoSource.resizeSource(0.33);
    videoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //craete grouping object
    tt::Grouping mGrouping;
    
#define USE_SCALE
    
#ifndef USE_SCALE
    tt::GH mGH(videoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GH_Arth_Perspective.dat";
#else
    tt::GHscale mGH(videoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GHscale_Arth_Perspective.dat";
#endif
    {
        //load perspective training GH
        std::ifstream geomHashingStream(GHfilename, std::ios::in | std::ios::binary);
        mGH.loadFromStream(geomHashingStream);
    }
    
    //process sequence
    Affine3d robotPose;
    bool found=false;
    while(1)
    {
        //get new image
        videoSource.grabNewFrame();
        Mat inputImage = videoSource.getFramePointer();
        
        //get the pairs which are likely to belong to group of blobs from model
        vector<KeyPoint> blobs;
        vector<tt::BlobPair> blobPairs;
        mGrouping.getBlobsAndPairs(inputImage,blobs,blobPairs);
        
        //get triplet by checking homography and inertia
        vector<tt::BlobTriplet> blobTriplets;
        mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
        
        //get only blobs found in triplets
        vector<KeyPoint> blobsinTriplets;
        tt::getBlobsInTriplets(blobs,blobTriplets,blobsinTriplets);
        
        //extract blobs and identify which one fit model, return set of positions and Id
        vector<tt::DetectionGH> mMatches;
        mGH.getModelPointsFromImage(blobsinTriplets,mMatches);
        //mGH.getModelPointsFromImage(inputImage,mMatches);
        
        //compute robots pose
        tt::ThymioBlobModel mRobot;
        found = mRobot.getPose(videoSource.mCalibration,mMatches,robotPose,!found);
        
        if(found)//draw model from found pose
            mRobot.draw(inputImage,videoSource.mCalibration, robotPose);
        else
            putText(inputImage, "Lost", Point2i(10,10),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
        
        
        //flush
        drawBlobPairs(inputImage,blobs,blobPairs);
        drawBlobTriplets(inputImage,blobs,blobTriplets);
        drawPointsAndIds(inputImage,mMatches);
        
        //check for rectangular blobs
        //extractAndDrawRectangularBlobs(inputImage);
        
        
        //process(inputImage);
        imshow(window_name,inputImage);
        
        //press escape to leave loop
        if(waitKey(5) == 27)break;
        //waitKey();
    }
    
}

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

void keyPointMatching()
{
    /// create display window
    //namedWindow( window_name, WINDOW_AUTOSIZE );

    //std::vector<KeyPoint> keypoints_1,keypoints_2;
    //SurfFeatureDetector detector1( hessianThreshold, octaves, octaveLayers, upright );
    
    Mat imageModel;
    imageModel = imread("/Users/amaurydame/Libs/ferns_demo-1.1/backCropped.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    //Mat inputImage;
    //inputImage = imread("/Users/amaurydame/Libs/ferns_demo-1.1/backRobot.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    //inputImage = imread("/Users/amaurydame/Libs/ferns_demo-1.1/backRobot2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    //FAST(imageModel, keypoints_1, 15);
    //FAST(inputImage, keypoints_2, 15);
    
    //Ptr<ORB> mORB;
    //mORB = ORB::create();
    //mORB = ORB::create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)
    //mORB = ORB::create(1000,1.2f,4,31,0,2, ORB::FAST_SCORE,31);
    //mORB->detect(inputImage,keypoints_2);
    
    //Ptr<BRISK> mBRISK;
    //mBRISK = BRISK::create();
    //mBRISK->detect(inputImage,keypoints_2);
    
    //Ptr<KAZE> mKAZE;
    //mKAZE = KAZE::create();
    //mKAZE->detect(inputImage,keypoints_2);
    
    /*Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(imageModel, noArray(), keypoints_1, desc1);
    akaze->detectAndCompute(inputImage, noArray(), keypoints_2, desc2);
    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    
    std::vector<DMatch> match1;
    std::vector<DMatch> match2;
    
    for(unsigned int i=0; i<nn_matches.size(); i++)
    {
        match1.push_back(nn_matches[i][0]);
        match2.push_back(nn_matches[i][1]);
    }
    
    //Mat img_matches1, img_matches2;
    //drawMatches(inputImage, keypoints_1, imageModel, keypoints_2, match1, img_matches1);
    
    for(unsigned int p=0;p<match1.size();p++)
        cv::circle(inputImage, keypoints_2[match1[p].trainIdx].pt, (keypoints_2[match1[p].trainIdx].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), 0);
    for(unsigned int p=0;p<match2.size();p++)
        cv::circle(inputImage, keypoints_2[match2[p].trainIdx].pt, (keypoints_2[match2[p].trainIdx].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), 0);
*/
    //drawMatches(imageModel, keypoints_1, inputImage, keypoints_2, match1, img_matches1);
    //drawMatches(imageModel, keypoints_1, inputImage, keypoints_2, match2, img_matches2);
    
    //detector1.detect( imageModel, keypoints_1 );
    //detector1.detect( inputImage, keypoints_2 );
    
    //for(unsigned int p=0;p<keypoints_2.size();p++)
    //    cv::circle(inputImage, keypoints_2[p].pt, (keypoints_2[p].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), 0);
    
    //imshow(window_name,inputImage);
    //waitKey();
    
    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    while(1)
    {
        //get new image
        videoSource.grabNewFrame();
        Mat inputImageCol = videoSource.getFramePointer();
        
        cv::Mat inputImage;
        cv::cvtColor(inputImageCol, inputImage, CV_BGR2GRAY);
        
        std::vector<KeyPoint> keypoints_1,keypoints_2;
        Mat desc1, desc2;
        Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(imageModel, noArray(), keypoints_1, desc1);
        akaze->detectAndCompute(inputImage, noArray(), keypoints_2, desc2);
        
        BFMatcher matcher(NORM_HAMMING);
        vector< vector<DMatch> > nn_matches;
        matcher.knnMatch(desc1, desc2, nn_matches, 2);
        
        std::vector<DMatch> match1;
        std::vector<DMatch> match2;
        
        for(unsigned int i=0; i<nn_matches.size(); i++)
        {
            match1.push_back(nn_matches[i][0]);
            match2.push_back(nn_matches[i][1]);
        }
        
        //Mat img_matches1, img_matches2;
        //drawMatches(inputImage, keypoints_1, imageModel, keypoints_2, match1, img_matches1);
        
        for(unsigned int p=0;p<match1.size();p++)
            cv::circle(inputImageCol, keypoints_2[match1[p].trainIdx].pt, (keypoints_2[match1[p].trainIdx].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), 0);
        for(unsigned int p=0;p<match2.size();p++)
            cv::circle(inputImageCol, keypoints_2[match2[p].trainIdx].pt, (keypoints_2[match2[p].trainIdx].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), 0);
        
        //process(inputImage);
        imshow(window_name,inputImageCol);
        
        //press escape to leave loop
        if(waitKey(5) == 27)break;
        //waitKey();
    }



}

int main( int argc, char** argv )
{
    //doGHmatching();
    //searchGoodPairs();
    GoodPairsAndGH();
    //keyPointMatching();
    return 0;

}

