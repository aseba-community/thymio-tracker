//Application to test matching by Geometric hashing

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//opencv display
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

void print_usage(const char* command)
{
    std::cerr << "Usage:\n\t" << command << " <geo hashing file>" << std::endl;
}


int main( int argc, char** argv )
{
    if(argc != 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    //output file, typically "../data/GHscale_Arth_Perspective.xml"
    std::string ghFilename = argv[1];

    //load camera and intrinsic parameters
    VideoSourceLive videoSource;
    videoSource.resizeSource(0.5);

    //camera calibration
    tt::IntrinsicCalibration mCalibration(tt::EmbeddedCam);

    /// create display window
    namedWindow( window_name, WINDOW_AUTOSIZE );
    
    //create grouping module: allow for checking if a blob is part 
    //of a valid pair of blobs, or triplet
    tt::Grouping mGrouping;
    
#define USE_SCALE
    
#ifndef USE_SCALE
    tt::GH mGH(&mCalibration);
#else
    tt::GHscale mGH(&mCalibration);
#endif

    //load perspective training GH
    cv::FileStorage GHstorage(ghFilename, cv::FileStorage::READ);
    if (GHstorage.isOpened())
        mGH.loadFromFileStorage(GHstorage);
    else
    {
        std::cerr<<"Problem loading GH xml file..."<<std::endl;
        return 1;
    }
     
    //process sequence
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
        
        //flush
        drawBlobPairs(inputImage,blobs,blobPairs);
        drawBlobTriplets(inputImage,blobs,blobTriplets);
        drawPointsAndIds(inputImage,mMatches);
                
        //process(inputImage);
        imshow(window_name,inputImage);
        
        //press escape to leave loop
        if(waitKey(5) == 27)
            break;
    }

        return 0;

    
}

