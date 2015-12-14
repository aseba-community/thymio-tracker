//Geometric hashing, first version: 2 dimensions hash table
//for each point: get nbNeigbor=3 closest points,
//consider all positively oriented triangles in this set of points
//create a 3D basis from that and project other points in it
#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "Generic.hpp"
#include "BlobInertia.hpp"

namespace thymio_tracker
{

struct BlobPair {
    int ids[2];//id of blobs in list of blobs
    BlobPair(int _id1,int _id2){ids[0]=_id1;ids[1]=_id2;};
} ;
struct BlobTriplet {
    int ids[3];
    BlobTriplet(){};
    BlobTriplet(int _id1,int _id2,int _id3){ids[0]=_id1;ids[1]=_id2;ids[2]=_id3;};
} ;
struct BlobQuadruplets {
    int ids[4];
    BlobQuadruplets(){};
    BlobQuadruplets(int _id1,int _id2,int _id3,int _id4){ids[0]=_id1;ids[1]=_id2;ids[2]=_id3;;ids[3]=_id4;};
} ;

void drawBlobPairs(cv::Mat &img, const std::vector<cv::KeyPoint> &blobs, const std::vector<BlobPair> &blobPairs);
void drawBlobTriplets(cv::Mat &img, const std::vector<cv::KeyPoint> &blobs, const std::vector<BlobTriplet> &blobTriplets);
void drawBlobQuadruplets(cv::Mat &img, const std::vector<cv::KeyPoint> &blobs, const std::vector<BlobQuadruplets> &blobQuadriplets);
void getBlobsInTriplets(const std::vector<cv::KeyPoint> &blobs, const std::vector<BlobTriplet> &blobTriplets,std::vector<cv::KeyPoint> &blobsinTriplets);


class Grouping
{
public:
    //constructor
    Grouping();
    
    //extract blobs and get good pairs
    void getBlobsAndPairs(cv::Mat &inputImage, std::vector<cv::KeyPoint> &blobs, std::vector<BlobPair> &blobPairs);
    //get triplets from pairs checking homography and inertia
    void getTripletsFromPairs(std::vector<cv::KeyPoint> &blobs, std::vector<BlobPair> &blobPairs, std::vector<BlobTriplet> &blobTriplets);
    //get quadriplets from triplets by checking overlap
    void getQuadripletsFromTriplets(std::vector<BlobTriplet> &blobTriplets,std::vector<BlobQuadruplets> &blobQuadriplets);


private:
    //maximum of neigbours considered to build pair
    int nbNeigboursMax;
    
    //blob extractor
    cv::Ptr<cv::SimpleBlobDetectorInertia> sbd;
    void extractBlobs(const cv::Mat& input, std::vector<cv::KeyPoint> &blobs) const;
    void getClosestNeigbors(int p, const std::vector<cv::KeyPoint>& mVerticesDes, std::vector<int>& idNeigbors) const;

};

}
