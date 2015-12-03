//Geometric hashing, first version: 2 dimensions hash table
//for each point: get nbNeigbor=3 closest points,
//consider all positively oriented triangles in this set of points
//create a 3D basis from that and project other points in it
#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "Generic.hpp"

namespace thymio_tracker
{

class GHscale
{
public:
    //constructor
    GHscale(IntrinsicCalibration _camCalib = IntrinsicCalibration());
    void setCalibration(IntrinsicCalibration &_camCalib){cameraCalibration=_camCalib;};
    virtual ~GHscale();
    
    //set Hashing table, could do that autonomously from model, but as we have one object we code it the hard way
    void initHashTable(int _nbIds, cv::Point3i _nbBinsPerDim=cv::Point3i(40,40,5));
    //train GH with model projected in several positions to be more robust to perspective effects
    void setModel(std::vector<cv::Point3f> *projPoints, int nbPoses);
    //extract blobs, get there 3D position, check which point they correspond to in HashTable
    void getModelPointsFromImage(const cv::Mat& img, std::vector<DetectionGH> &matches) const;
    void getModelPointsFromImage(const std::vector<cv::KeyPoint> &blobs, std::vector<DetectionGH> &matches) const;

    //GH io
    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);

private:
    //camera calibration
    IntrinsicCalibration cameraCalibration;
    //blob extractor
    cv::Ptr<cv::SimpleBlobDetector> sbd;
    //extract the blob position and scales for getModelPointsFromImage
    void extractBlobs(const cv::Mat& input, std::vector<cv::KeyPoint> &blobs) const;
    //unproject blobs from image space to "world space"
    //void convertToWorldFrame(vector<KeyPoint> &blobs,Mat &cameraMatrix, Mat &distCoeffs,vector<Point3f> &points3d);
    //get the nbPtBasis closest points to p
    void getClosestNeigbors(int p, const std::vector<cv::Point3f>& mVerticesDes, std::vector<int>& idNeigbors) const;
    //made to do some testing: compute hashTable corresponding to a special base and save data to display
    //void getSignatureBasis(vector<Point3f> &mVerticesDes, vector<int> &basisId, char *filename);
    //smoothes votes in HastTable: indeed current base will differ from model base due to measurement erros => if many bins might read votes in one bin that is just neigboring the one we actually want to read. Can also allow for perspective distortion if depth blobs are omitted
    void blurHashTable();


    //attributes
    //HashTable: each cell stores a number of vote for each point id
    int nbIds;
    cv::Point3i nbBinPerDim;
    float *HashTable;//binx first, biny, binscale, id
    
    //function to navigate in HT:
    cv::Point3f poseRelMin, poseRelMax;
    cv::Point3f toCell(const cv::Point3f& relativePos) const;
    //add some votes _v (eg 1 for one vote) in bin bin for point id. (for training)
    void addVoteToBin(const cv::Point3f& bin,const int &id, const float _v);
    //get the votes for each id corresponding to one bin. (for matching)
    void readVotesFromBin(const cv::Point3f& bin,float *votes) const;
    
    //number of neigbors considered for each point to define bases
    int nbPtBasis;
};

}
