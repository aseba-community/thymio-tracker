//matching by Geometric hashing

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//opencv display
//Mat inputImage;
char window_name[] = "InputWindow";

//3D model, first simple version with just top
#include "VideoSource.hpp"
#include "GH.hpp"
#include "GHscale.hpp"
#include "Visualization3D.hpp"
#include "Grouping.hpp"



void drawPointsAndIds(Mat &inputImage,vector<DetectionGH> &_matches)
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

void doGHmatching()
{
    //load camera and intrinsic parameters
    //videoSourceSeq mVideoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    videoSourceLive mVideoSource(EmbeddedCam);
    
    //resize input
    //mVideoSource.resizeSource(0.33);
    mVideoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );

#define USE_SCALE

#ifndef USE_SCALE
    GH mGH(mVideoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GH_Arth_Perspective.dat";
#else
    GHscale mGH(mVideoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GHscale_Arth_Perspective.dat";
#endif
    //load perspective training GH
    mGH.loadFromFile(GHfilename);
 
    //process sequence
    Affine3d robotPose;
    bool found=false;
    while(1)
    {
        //get new image
        mVideoSource.grabNewFrame();
        Mat &inputImage=*mVideoSource.GetFramePointer();

        clock_t startTime = clock();
        
        //extract blobs and identify which one fit model, return set of positions and Id
        vector<DetectionGH> mMatches;
        mGH.getModelPointsFromImage(inputImage,mMatches);
        
        //compute robots pose
        ArthymioBlobModel mRobot;
        found = mRobot.getPose(mVideoSource.mCalibration,mMatches,robotPose,!found);
        
        cout << double( 1000.*(clock() - startTime) ) / (double)CLOCKS_PER_SEC<< " ms." << endl;
        
        if(found)//draw model from found pose
            mRobot.draw(inputImage,mVideoSource.mCalibration, robotPose);
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

void drawBlobPairs(Mat &img,vector<KeyPoint> &blobs, vector<BlobPair> &blobPairs)
{
    //draw Blobs
    for(int p=0;p<blobs.size();p++)
        cv::circle(img, blobs[p].pt, (blobs[p].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
    //draw Pairs
    for(int i=0;i<blobPairs.size();i++)
        line(img, blobs[blobPairs[i].ids[0]].pt, blobs[blobPairs[i].ids[1]].pt, Scalar(0,0,255), 5);
}

void drawBlobTriplets(Mat &img,vector<KeyPoint> &blobs, vector<BlobTriplet> &blobTriplets)
{
    //draw Triplets
    for(int i=0;i<blobTriplets.size();i++)
    {
        line(img, blobs[blobTriplets[i].ids[0]].pt, blobs[blobTriplets[i].ids[1]].pt, Scalar(155,0,155), 3);
        line(img, blobs[blobTriplets[i].ids[1]].pt, blobs[blobTriplets[i].ids[2]].pt, Scalar(155,0,155), 3);
        line(img, blobs[blobTriplets[i].ids[2]].pt, blobs[blobTriplets[i].ids[0]].pt, Scalar(155,0,155), 3);
    }
}

void drawBlobQuadruplets(Mat &img,vector<KeyPoint> &blobs, vector<BlobQuadruplets> &blobQuadriplets)
{
    //draw Triplets
    for(int i=0;i<blobQuadriplets.size();i++)
    {
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[1]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[2]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[1]].pt, blobs[blobQuadriplets[i].ids[2]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[1]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[2]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
    }
}

void searchGoodPairs()
{
    //load camera and intrinsic parameters
    //videoSourceSeq mVideoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    videoSourceLive mVideoSource(EmbeddedCam);
    
    //resize input
    //mVideoSource.resizeSource(0.33);
    mVideoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //craete grouping object
    Grouping mGrouping;
    while(1)
    {
        //get new image
        mVideoSource.grabNewFrame();
        Mat &inputImage=*mVideoSource.GetFramePointer();
        
        //get the pairs which are likely to belong to group of blobs from model
        vector<KeyPoint> blobs;
        vector<BlobPair> blobPairs;
        mGrouping.getBlobsAndPairs(inputImage,blobs,blobPairs);
        
        //get triplet by checking homography and inertia
        vector<BlobTriplet> blobTriplets;
        mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
        
        vector<BlobQuadruplets> blobQuadriplets;
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

void getBlobsInTriplets(const vector<KeyPoint> &blobs,const vector<BlobTriplet> &blobTriplets,vector<KeyPoint> &blobsinTriplets)
{
    //get all the ids of the blobs in the triplets, taking care of duplicates
    vector<int> idBlobsInTripelts;
    for(int i=0;i<blobTriplets.size();i++)
    {
        for(int t=0;t<3;t++)
        {
            int idc=blobTriplets[i].ids[t];
            if(find(idBlobsInTripelts.begin(), idBlobsInTripelts.end(), idc)==idBlobsInTripelts.end())
                idBlobsInTripelts.push_back(idc);
        }
    }
    //create the new blob vector
    for(int i=0;i<idBlobsInTripelts.size();i++)
        blobsinTriplets.push_back(blobs[idBlobsInTripelts[i]]);
    //blobsinTriplets=blobs;
}

void GoodPairsAndGH()
{
    //load camera and intrinsic parameters
    //videoSourceSeq mVideoSource("/Users/amaurydame/Data/nexus/TrackSeq2/image-%03d.png",NexusCam,1);
    videoSourceLive mVideoSource(EmbeddedCam);
    
    //resize input
    //mVideoSource.resizeSource(0.33);
    mVideoSource.resizeSource(0.5);
    
    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //craete grouping object
    Grouping mGrouping;
    
#define USE_SCALE
    
#ifndef USE_SCALE
    GH mGH(mVideoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GH_Arth_Perspective.dat";
#else
    GHscale mGH(mVideoSource.mCalibration);
    char GHfilename[100]="/Users/amaurydame/Projects/BlobotTracker/files/GHscale_Arth_Perspective.dat";
#endif
    //load perspective training GH
    mGH.loadFromFile(GHfilename);
    
    //process sequence
    Affine3d robotPose;
    bool found=false;
    while(1)
    {
        //get new image
        mVideoSource.grabNewFrame();
        Mat &inputImage=*mVideoSource.GetFramePointer();
        
        //get the pairs which are likely to belong to group of blobs from model
        vector<KeyPoint> blobs;
        vector<BlobPair> blobPairs;
        mGrouping.getBlobsAndPairs(inputImage,blobs,blobPairs);
        
        //get triplet by checking homography and inertia
        vector<BlobTriplet> blobTriplets;
        mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
        
        //get only blobs found in triplets
        vector<KeyPoint> blobsinTriplets;
        getBlobsInTriplets(blobs,blobTriplets,blobsinTriplets);
        
        //extract blobs and identify which one fit model, return set of positions and Id
        vector<DetectionGH> mMatches;
        mGH.getModelPointsFromImage(blobsinTriplets,mMatches);
        //mGH.getModelPointsFromImage(inputImage,mMatches);
        
        //compute robots pose
        ArthymioBlobModel mRobot;
        found = mRobot.getPose(mVideoSource.mCalibration,mMatches,robotPose,!found);
        
        if(found)//draw model from found pose
            mRobot.draw(inputImage,mVideoSource.mCalibration, robotPose);
        else
            putText(inputImage, "Lost", Point2i(10,10),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
        
        
        //flush
        drawBlobPairs(inputImage,blobs,blobPairs);
        drawBlobTriplets(inputImage,blobs,blobTriplets);
        drawPointsAndIds(inputImage,mMatches);
        
        
        //process(inputImage);
        imshow(window_name,inputImage);
        
        //press escape to leave loop
        if(waitKey(5) == 27)break;
        //waitKey();
    }
    
}

int main( int argc, char** argv )
{
    //doGHmatching();
    searchGoodPairs();
    //GoodPairsAndGH();
    return 0;

}