
#include "Grouping.hpp"

#include <stdexcept>

#include <opencv2/calib3d.hpp> //solvePnP
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace thymio_tracker
{

Grouping::Grouping()
{
    nbNeigboursMax=5;
    
    cv::SimpleBlobDetectorInertia::Params params;
    params.thresholdStep = 10;
    params.minThreshold = 40;
    params.maxThreshold = 210;
    params.minDistBetweenBlobs = 4;
    params.minRepeatability = 2;
    
    params.filterByColor = true;
    params.blobColor = 0;
    
    params.filterByArea = true;
    params.minArea = 5;
    params.maxArea = 800;
    
    params.filterByCircularity = true;
    params.minCircularity = 0.8;
    //params.minCircularity = 0.5;
    params.maxCircularity = 1.4;
    
    params.filterByInertia = true;
    params.minInertiaRatio = 0.1;
    params.minInertiaRatio = 0.3;
    params.maxInertiaRatio = 1.0;
    
    params.filterByConvexity = false;
    sbd = cv::SimpleBlobDetectorInertia::create(params);
    
}

void Grouping::extractBlobs(const cv::Mat& input, vector<KeyPoint> &blobs) const
{
    Mat gray;
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    
    // blob detector
    sbd->detect(gray, blobs);
}

struct sort_wrt_second {
    bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
        return left.second < right.second;
    }
};

void Grouping::getClosestNeigbors(unsigned int p, const vector<KeyPoint>& mVerticesDes, vector<unsigned int>& idNeigbors) const
{
    //create pairs of point indexes and corresponding distance and sort with respect to deistance
    vector< pair<int,float> > pairIdDist;
    //only use keypoints after p to not have duplicated pairs
    for(unsigned int i=p+1;i<mVerticesDes.size();i++)
    {
        pair<int,float> newPair;
        newPair.first=i;
        newPair.second=norm(mVerticesDes[i].pt-mVerticesDes[p].pt);
        pairIdDist.push_back(newPair);
    }
    
    //sort it
    std::sort(pairIdDist.begin(), pairIdDist.end(), sort_wrt_second());
    
    //return first elements
    for(unsigned int i=0;i<pairIdDist.size() && i<nbNeigboursMax;i++)
        idNeigbors.push_back(pairIdDist[i].first);
    
}

void Grouping::getBlobs(cv::Mat &img, std::vector<cv::KeyPoint> &blobs)
{
    //get blobs
    extractBlobs(img, blobs);
}
void Grouping::getPairsFromBlobs(const std::vector<cv::KeyPoint> &blobs, std::vector<BlobPair> &blobPairs)
{
    for(int p=0;p<blobs.size();p++)
    {
        //for each point have to find the nbPtBasis closest points
        vector<unsigned int> idNeigbors;
        //only get neigbours indexed after p
        getClosestNeigbors(p, blobs, idNeigbors);
        
        //check resulting pairs
        for(int i=0;i<idNeigbors.size();i++)
        {
            float d_on_ss=norm(blobs[p].pt-blobs[idNeigbors[i]].pt)/sqrt(blobs[p].size*blobs[idNeigbors[i]].size);
            float scale_dist=sqrt((blobs[p].size-blobs[idNeigbors[i]].size)*(blobs[p].size-blobs[idNeigbors[i]].size));
            //check with respect to stat we got in blobStat
            //if(d_on_ss>1.4 && d_on_ss<2.1)//ratio distance/scale is good, then add pair
            if(d_on_ss>1. && d_on_ss<3.)//ratio distance/scale is good, then add pair
                if(scale_dist<2)//pairs of blobs are close so scale diff should not be too big
                    blobPairs.push_back(BlobPair(p,idNeigbors[i]));
        }
    }
}

void Grouping::getBlobsAndPairs(const cv::Mat &img, std::vector<cv::KeyPoint> &blobs, std::vector<BlobPair> &blobPairs)
{
    //get blobs
    extractBlobs(img, blobs);
    
    for(unsigned int p=0;p<blobs.size();p++)
    {
        //for each point have to find the nbPtBasis closest points
        vector<unsigned int> idNeigbors;
        //only get neigbours indexed after p
        getClosestNeigbors(p, blobs, idNeigbors);
        
        //check resulting pairs
        for(unsigned int i=0;i<idNeigbors.size();i++)
        {
            float d_on_ss=norm(blobs[p].pt-blobs[idNeigbors[i]].pt)/sqrt(blobs[p].size*blobs[idNeigbors[i]].size);
            float scale_dist=sqrt((blobs[p].size-blobs[idNeigbors[i]].size)*(blobs[p].size-blobs[idNeigbors[i]].size));
            //check with respect to stat we got in blobStat
            //if(d_on_ss>1.4 && d_on_ss<2.1)//ratio distance/scale is good, then add pair
            if(d_on_ss>1. && d_on_ss<3.)//ratio distance/scale is good, then add pair
                if(scale_dist<2)//pairs of blobs are close so scale diff should not be too big
                    blobPairs.push_back(BlobPair(p,idNeigbors[i]));
        }
    }
}

void Grouping::getTripletsFromPairs(const std::vector<cv::KeyPoint> &blobs, std::vector<BlobPair> &blobPairs, std::vector<BlobTriplet> &blobTriplets)
{
    for(unsigned int p=0;p<blobPairs.size();p++)
    {
        //get the pairs which overlap
        //note that in pair we always have id1 < id2
        for(unsigned int p2=p+1;p2<blobPairs.size();p2++)
        {
            BlobTriplet newTriplet;
        
            //check if overlap
            if(blobPairs[p].ids[0]==blobPairs[p2].ids[0])
            {
                newTriplet.ids[0]=blobPairs[p].ids[0];newTriplet.ids[1]=blobPairs[p].ids[1];newTriplet.ids[2]=blobPairs[p2].ids[1];
            }
            else if(blobPairs[p].ids[0]==blobPairs[p2].ids[1])
            {
                newTriplet.ids[0]=blobPairs[p].ids[0];newTriplet.ids[1]=blobPairs[p].ids[1];newTriplet.ids[2]=blobPairs[p2].ids[0];
            }
            else if(blobPairs[p].ids[1]==blobPairs[p2].ids[0])
            {
                newTriplet.ids[0]=blobPairs[p].ids[1];newTriplet.ids[1]=blobPairs[p].ids[0];newTriplet.ids[2]=blobPairs[p2].ids[1];
            }
            else if(blobPairs[p].ids[1]==blobPairs[p2].ids[1])
            {
                newTriplet.ids[0]=blobPairs[p].ids[1];newTriplet.ids[1]=blobPairs[p].ids[0];newTriplet.ids[2]=blobPairs[p2].ids[0];
            }
            else
                continue;
            
            
            
            //check if their is an homography which fits and respect inertia
            vector<KeyPoint> blobPoints;
            blobPoints.push_back(blobs[newTriplet.ids[0]]);
            blobPoints.push_back(blobs[newTriplet.ids[1]]);
            blobPoints.push_back(blobs[newTriplet.ids[2]]);
            
            for(int m=0;m<3;m++)//for all the possible arrangement of the points
            {
                //get arranged points
                vector<Point2f> blobPointsArranged;
                blobPointsArranged.push_back(blobPoints[m % 3].pt);  //0 0 1 1 2 2
                blobPointsArranged.push_back(blobPoints[(m+1) % 3].pt);  //1 2 0 2 0 1
                blobPointsArranged.push_back(blobPoints[(m+2) % 3].pt);  //2 1 2 0 1 0
                
                //just check inertia
                //basis vectors:
                Point2f v1=blobPointsArranged[1]-blobPointsArranged[0];
                Point2f v2=blobPointsArranged[2]-blobPointsArranged[0];
                
                //normalize
                float maxNorm=(norm(v1)>norm(v2))?norm(v1):norm(v2);
                v1 = v1/maxNorm;
                v2 = v2/maxNorm;
                
                //inetria of blobs should be related to area formed by v1 and v2
                //get the area with determinant
                //Keypoint resulting from blob has been modified to output its inertia in the response attribute
                float inertia_des = v1.x*v2.y-v1.y*v2.x;
                if(inertia_des<0)inertia_des=-inertia_des;
                
                float inertia_error=0;
                for(unsigned int i=0;i<blobPoints.size();i++)
                {
                    inertia_error+=sqrt((blobPoints[i].response-inertia_des)*(blobPoints[i].response-inertia_des));
                    //cout<<"blobPoints["<<i<<"].response = "<<blobPoints[i].response<<"  inertia des = "<<inertia_des<<endl;
                }
                inertia_error=inertia_error/blobPoints.size();
                //cout<<"error = "<<inertia_error<<endl;
                
                if(inertia_error<0.3 && //want the inertia to match perspective transfo
                   inertia_des>0.2) //want to have a solution that is feasible (if inertia_des is too smal then wont detect blobs
                {
                    blobTriplets.push_back(newTriplet);
                    break;
                }
                
            }
        }           
    }
}

void Grouping::getQuadripletsFromTriplets(std::vector<BlobTriplet> &blobTriplets,std::vector<BlobQuadruplets> &blobQuadriplets,bool removeTripletsInQuads)
{
    //many ways to do, for now go through list of triplets and check if shares 2 points with other triangles,
    //if doesn't remove it from list, if does create quadruplets and remove all other triangle contained in quadruplets
    std::vector<BlobTriplet> blobTripletsCopy=blobTriplets;

    //to keep track of the id of the triplets in the original triplets list
    std::vector<int> idAllTripletsInCopy;
    for(int i=0;i<blobTripletsCopy.size();i++)idAllTripletsInCopy.push_back(i);

    //to store the id of the triplets in quads
    std::vector<int> idTripletsInQuads;
    
    while(blobTripletsCopy.size()>0)
    {
        bool found=false;
        for(unsigned int t=1;t<blobTripletsCopy.size();t++)
        {
            //count how many points first triplet has in common with triplet[t]
            int pt_in_common=0;
            for(int i=0;i<3;i++)
                for(int i2=0;i2<3;i2++)
                    if(blobTripletsCopy[0].ids[i]==blobTripletsCopy[t].ids[i2])
                        pt_in_common++;
            
            if (pt_in_common==2) {
                found=true;
                
                //create quadruplet
                BlobQuadruplets newQuadruplets;
                for(int i=0;i<3;i++)newQuadruplets.ids[i]=blobTripletsCopy[0].ids[i];
                for(int i2=0;i2<3;i2++)//search for id in blobTriplets[t] too add
                {
                    bool bfound=false;
                    for(int i=0;i<3;i++)
                        if(blobTripletsCopy[0].ids[i]==blobTripletsCopy[t].ids[i2])bfound=true;
                    
                    if(!bfound)
                    {
                        newQuadruplets.ids[3]=blobTripletsCopy[t].ids[i2];
                        break;
                    }
                }
                //add quadruplet
                blobQuadriplets.push_back(newQuadruplets);
                
                //store id of found triplets
                idTripletsInQuads.push_back(idAllTripletsInCopy[0]);
                idTripletsInQuads.push_back(idAllTripletsInCopy[t]);


                //remove triplets 0 and t
                blobTripletsCopy.erase(blobTripletsCopy.begin()+t);
                blobTripletsCopy.erase(blobTripletsCopy.begin());

                idAllTripletsInCopy.erase(idAllTripletsInCopy.begin()+t);
                idAllTripletsInCopy.erase(idAllTripletsInCopy.begin());
                
                //search for all the other triplets included in quadruplet
                for(int t2=blobTripletsCopy.size()-1;t2>=0;t2--)
                {
                    //count how many points first triplet has in common with triplet[t]
                    int pt_in_common2=0;
                    for(int i=0;i<4;i++)
                        for(int i2=0;i2<3;i2++)
                            if(newQuadruplets.ids[i]==blobTripletsCopy[t2].ids[i2])
                                pt_in_common2++;
                    
                    if(pt_in_common2==3)
                    {
                        idTripletsInQuads.push_back(idAllTripletsInCopy[t2]);
                        blobTripletsCopy.erase(blobTripletsCopy.begin()+t2);
                        idAllTripletsInCopy.erase(idAllTripletsInCopy.begin()+t2);
                    }
                 }
                
                break;
            }                
        }
        
        //if first triplet was not found then remove it
        if(!found)
        {
            blobTripletsCopy.erase(blobTripletsCopy.begin());
            idAllTripletsInCopy.erase(idAllTripletsInCopy.begin());
        }
    }

    //if want to suppress the triplets in the quads then do so
    if(removeTripletsInQuads)
    {
        std::sort(idTripletsInQuads.begin(), idTripletsInQuads.end());
        for(int i=idTripletsInQuads.size()-1;i>=0;i--)
            blobTriplets.erase(blobTriplets.begin()+idTripletsInQuads[i]);
    }
}

void setClockwiseDirectionToTriplets(const std::vector<cv::KeyPoint> &blobs,vector<BlobTriplet> &blobTriplets)
{
    for(int i=0;i<blobTriplets.size();i++)
    {
        BlobTriplet &tp = blobTriplets[i];
        Point2f basis1 = blobs[tp.ids[1]].pt-blobs[tp.ids[0]].pt;
        Point2f basis2 = blobs[tp.ids[2]].pt-blobs[tp.ids[0]].pt;
        if(!testDirectionBasis(basis1,basis2))
        {
            int buff = tp.ids[2];
            tp.ids[2]=tp.ids[1];
            tp.ids[1]=buff;
        }
            
    }
    
}
    
void setClockwiseDirectionToQuadruplets(const std::vector<cv::KeyPoint> &blobs,vector<BlobQuadruplets> &blobQuadruplets)
{
    //want quadruplet ordered clockwise, triplet from first indexes should already be set accordingly
    //=> now want to know if we have to move last index to second or third position
    
    for(int i=0;i<blobQuadruplets.size();i++)
    {
        BlobQuadruplets &tp = blobQuadruplets[i];
        int unsortedIndex = tp.ids[3];
        Point2f basis4 = blobs[tp.ids[3]].pt-blobs[tp.ids[0]].pt;
        
        Point2f basis2 = blobs[tp.ids[1]].pt-blobs[tp.ids[0]].pt;
        Point2f basis3 = blobs[tp.ids[2]].pt-blobs[tp.ids[0]].pt;
        
        bool b2Right = testDirectionBasis(basis2,basis4);
        if(!b2Right)
        {
            tp.ids[3]=tp.ids[2];
            tp.ids[2]=tp.ids[1];
            tp.ids[1]=unsortedIndex;
        }
        else
        {
            bool b3Right = testDirectionBasis(basis3,basis4);
            if(!b3Right)
            {
                tp.ids[3]=tp.ids[2];
                tp.ids[2]=unsortedIndex;
            }
        }
        
    }
    
}

void drawBlobPairs(Mat &img, const vector<KeyPoint> &blobs, const vector<BlobPair> &blobPairs)
{
    //draw Blobs
    for(unsigned int p=0;p<blobs.size();p++)
        cv::circle(img, blobs[p].pt, (blobs[p].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
    //draw Pairs
    for(unsigned int i=0;i<blobPairs.size();i++)
        line(img, blobs[blobPairs[i].ids[0]].pt, blobs[blobPairs[i].ids[1]].pt, Scalar(0,0,255), 5);
}

void drawBlobTriplets(Mat &img, const vector<KeyPoint> &blobs, const vector<BlobTriplet> &blobTriplets)
{
    //draw Triplets
    for(unsigned int i=0;i<blobTriplets.size();i++)
    {
        line(img, blobs[blobTriplets[i].ids[0]].pt, blobs[blobTriplets[i].ids[1]].pt, Scalar(155,0,155), 3);
        line(img, blobs[blobTriplets[i].ids[1]].pt, blobs[blobTriplets[i].ids[2]].pt, Scalar(155,0,155), 3);
        line(img, blobs[blobTriplets[i].ids[2]].pt, blobs[blobTriplets[i].ids[0]].pt, Scalar(155,0,155), 3);
    }
}

void drawBlobQuadruplets(Mat &img, const vector<KeyPoint> &blobs, const vector<BlobQuadruplets> &blobQuadriplets)
{
    //draw Triplets
    for(unsigned int i=0;i<blobQuadriplets.size();i++)
    {
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[1]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[2]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[0]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[1]].pt, blobs[blobQuadriplets[i].ids[2]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[1]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
        line(img, blobs[blobQuadriplets[i].ids[2]].pt, blobs[blobQuadriplets[i].ids[3]].pt, Scalar(0,255,255), 2);
    }
}
void getBlobsInTriplets(const vector<KeyPoint> &blobs,const vector<BlobTriplet> &blobTriplets,vector<KeyPoint> &blobsinTriplets)
{
    //get all the ids of the blobs in the triplets, taking care of duplicates
    vector<int> idBlobsInTripelts;
    for(unsigned int i=0;i<blobTriplets.size();i++)
    {
        for(int t=0;t<3;t++)
        {
            int idc=blobTriplets[i].ids[t];
            if(find(idBlobsInTripelts.begin(), idBlobsInTripelts.end(), idc)==idBlobsInTripelts.end())
                idBlobsInTripelts.push_back(idc);
        }
    }
    //create the new blob vector
    for(unsigned int i=0;i<idBlobsInTripelts.size();i++)
        blobsinTriplets.push_back(blobs[idBlobsInTripelts[i]]);
    //blobsinTriplets=blobs;
}

}
