
#include "arthymio.h"

#include "GHscale.hpp"
#include "Models.hpp"

#include <vector>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Grouping.hpp"

static bool initialized = false;
static bool mFound = false;
static GHscale mGH;
static ArthymioBlobModel mRobot;
static IntrinsicCalibration calibration;
static float mScale = 1.0;
static Grouping mGrouping;

void drawPointsAndIds(cv::Mat& inputImage, const std::vector<DetectionGH>& matches)
{
    //draw Id
    for(int i=0;i<matches.size();i++)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", matches[i].id);
        circle(inputImage, matches[i].position, 4, cvScalar(0,250,250), -1, 8, 0);
        putText(inputImage, pointIdStr, matches[i].position, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}

void loadCalibration(const std::string& filename, IntrinsicCalibration* calibration, const cv::Size* imgSize)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }
    
    readCalibrationFromFileStorage(fs,*calibration);
    rescaleCalibration(*calibration, *imgSize);
    
    fs.release();
}

void init(const cv::Size &imgSize)
{
     static const std::string ghfilename = "../data/GHscale_Arth_Perspective.dat";
    //static const std::string ghfilename = "/sdcard/GH_Arth_Perspective.dat";
    mGH.loadFromFile(ghfilename);
    
    loadCalibration("../data/calibration/embedded_camera_calib.xml", &calibration, &imgSize);
    //loadCalibration("../data/calibration/nexus_camera_calib.xml", &calibration, &imgSize);
    ////loadCalibration("/sdcard/nexus_camera_calib.xml", &calibration, &imgSize, mScale);
    mGH.setCalibration(calibration);
    
    initialized = true;
}

int process(const cv::Mat& input, cv::Mat& output)
{
    // input.copyTo(output);
    cv::resize(input, output, cv::Size(0, 0), mScale, mScale);

    if(!initialized)
        init(output.size());
    
    
    cv::Affine3d robotPose;
    std::vector<DetectionGH> matches;
    
    //get the pairs which are likely to belong to group of blobs from model
    std::vector<cv::KeyPoint> blobs;
    std::vector<BlobPair> blobPairs;
    mGrouping.getBlobsAndPairs(output,blobs,blobPairs);

    //get triplet by checking homography and inertia
    std::vector<BlobTriplet> blobTriplets;
    mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
    
    //get only blobs found in triplets
    std::vector<cv::KeyPoint> blobsinTriplets;
    getBlobsInTriplets(blobs,blobTriplets,blobsinTriplets);
    
    std::vector<BlobQuadruplets> blobQuadriplets;
    mGrouping.getQuadripletsFromTriplets(blobTriplets,blobQuadriplets);

    //extract blobs and identify which one fit model, return set of positions and Id
    std::vector<DetectionGH> mMatches;
    mGH.getModelPointsFromImage(blobsinTriplets,mMatches);
    
    mFound = mRobot.getPose(calibration, matches, robotPose, !mFound);
    if(mFound)
        mRobot.draw(output, calibration, robotPose);
    else
        putText(output, "Lost", cv::Point2i(10,10),cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
    
    //flush
    drawBlobPairs(output,blobs,blobPairs);
    drawBlobTriplets(output,blobs,blobTriplets);
    drawBlobQuadruplets(output,blobs,blobQuadriplets);
    //drawPointsAndIds(output, matches);
    
    return 0;
}

// int get_rows(const cv::Mat& m)
// {
//     return m.rows;
// }
