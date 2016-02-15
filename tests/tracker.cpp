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
        
        //for testing direction groups:
        /*float size_test = 10;
        float inertia = 1.;
        blobs.push_back(cv::KeyPoint(Point2f(10,10),size_test,0,inertia));
        blobs.push_back(cv::KeyPoint(Point2f(30,10),size_test,0,inertia));
        blobs.push_back(cv::KeyPoint(Point2f(10,30),size_test,0,inertia));*/
        //blobs.push_back(cv::KeyPoint(Point2f(30,30),size_test,0,inertia));
        
        
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


