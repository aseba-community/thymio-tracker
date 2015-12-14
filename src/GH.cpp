
#include "GH.hpp"

#include <stdexcept>

#include <opencv2/calib3d.hpp> //solvePnP
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace thymio_tracker
{

GH::GH(IntrinsicCalibration _camCalib)
{
    cameraCalibration=_camCalib;
    //for now we will consider 3 neigboring points to define bases
    nbPtBasis=2;
    
    cv::SimpleBlobDetector::Params params;
    params.thresholdStep = 10;
    params.minThreshold = 50;
    params.maxThreshold = 200;
    params.minDistBetweenBlobs = 2;
    params.minRepeatability = 2;
    
    params.filterByColor = true;
    params.blobColor = 0;
    
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 5000;
    
    params.filterByCircularity = true;
    params.minCircularity = 0.8;
    params.maxCircularity = 1.2;
    
    params.filterByInertia = true;
    params.minInertiaRatio = 0.7;
    params.maxInertiaRatio = 1.3;
    
    params.filterByConvexity = false;
    sbd = cv::SimpleBlobDetector::create(params);
}

GH::~GH()
{
    delete[] HashTable;
}

//set Hashing table, could do that autonomously from model, but as we have one object we code it the hard way
void GH::initHashTable(int _nbIds, Point2i _nbBinsPerDim)
{
    nbIds=_nbIds;
    nbBinPerDim=_nbBinsPerDim;
    HashTable= new int[nbBinPerDim.x*nbBinPerDim.y*nbIds];
    
    //init all bins to 0 votes
    for(int i=0;i<nbBinPerDim.x*nbBinPerDim.y*nbIds;i++)
        HashTable[i]=0;
}

Point2i GH::toCell(const Point2f& relativePos) const
{
    Point2i res;
    res.x=(float)nbBinPerDim.x*(relativePos.x-poseRelMin.x)/(poseRelMax.x-poseRelMin.x);
    res.y=(float)nbBinPerDim.y*(relativePos.y-poseRelMin.y)/(poseRelMax.y-poseRelMin.y);
    return res;
}

struct sort_wrt_second {
    bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
        return left.second < right.second;
    }
};

void GH::getClosestNeigbors(int p, const vector<Point2f>& mVerticesDes, vector<int>& idNeigbors) const
{
    //create pairs of point indexes and corresponding distance and sort with respect to deistance
    vector< pair<int,float> > pairIdDist;
    for(int i=0;i<mVerticesDes.size();i++)
        if(i!=p)
    {
        pair<int,float> newPair;
        newPair.first=i;
        newPair.second=norm(mVerticesDes[i]-mVerticesDes[p]);
        pairIdDist.push_back(newPair);
    }
    
    //sort it
    std::sort(pairIdDist.begin(), pairIdDist.end(), sort_wrt_second());
    
    //return first elements
    for(int i=0;i<pairIdDist.size() && i<nbPtBasis;i++)
        idNeigbors.push_back(pairIdDist[i].first);
    
}


void GH::setModel(vector<Point2f> *projPoints, int nbPoses)
{
    //get hash table limits
    poseRelMin = Point2f(0,0); poseRelMax = Point2f(0,0);
    //get all the bases and project all point and set limits accordingly to support
    for(int idpose=0;idpose<nbPoses;idpose++)
    {
        vector<Point2f> &mProjs=projPoints[idpose];
        //loop through all points
        for(int p=0;p<mProjs.size();p++)
        {
            //for each point have to find the nbPtBasis closest points
            vector<int> idNeigbors;
            getClosestNeigbors(p,mProjs,idNeigbors);
            
            
            //for each positively oriented possible triangle in closest neigbors
            //define basis and project all points on it to fill HT
            for(int tp2=0;tp2<idNeigbors.size();tp2++)
            {
                //get index of point 2 in mVerticesDes
                int p2=idNeigbors[tp2];
                //define first basis vector
                Point2f basis1= mProjs[p2]-mProjs[p];
                
                for(int tp3=0;tp3<idNeigbors.size();tp3++)
                    if(p2!=idNeigbors[tp3])
                    {
                        int p3=idNeigbors[tp3];
                        //define second basis
                        Point2f basis2= mProjs[p3]-mProjs[p];
                        
                        //check direction of triangle
                        if(testDirectionBasis(basis1,basis2))
                        {
                            //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                            Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                            Matx22f tBasisInv=tBasis.inv();
                            
                            //good basis => project all points and fill HT
                            for(int i=0;i<mProjs.size();i++)
                                if(i!=p && i!=p2 && i!=p3)
                                {
                                    //project in current basis
                                    Point2f relCoord=tBasisInv*(mProjs[i]-mProjs[p]);
                                    
                                    //readjust limits
                                    if(relCoord.x<poseRelMin.x)poseRelMin.x=relCoord.x;
                                    if(relCoord.y<poseRelMin.y)poseRelMin.y=relCoord.y;
                                    if(relCoord.x>poseRelMax.x)poseRelMax.x=relCoord.x;
                                    if(relCoord.y>poseRelMax.y)poseRelMax.y=relCoord.y;
                                }
                            
                        }
                    }
                
            }
        }
    }
    //add margins
    float marginRel=0.1;
    float diffRellx=poseRelMax.x-poseRelMin.x;
    float diffRelly=poseRelMax.y-poseRelMin.y;
    poseRelMin.x=poseRelMin.x-marginRel*diffRellx;
    poseRelMin.y=poseRelMin.y-marginRel*diffRelly;
    poseRelMax.x=poseRelMax.x+marginRel*diffRellx;
    poseRelMax.y=poseRelMax.y+marginRel*diffRelly;

    for(int idpose=0;idpose<nbPoses;idpose++)
    {
        vector<Point2f> &mProjs = projPoints[idpose];
    
        //loop through all points to fill HT
        for(int p=0;p<mProjs.size();p++)
        {
            //for each point have to find the nbPtBasis closest points
            vector<int> idNeigbors;
            getClosestNeigbors(p,mProjs,idNeigbors);
            
            
            //for each positively oriented possible triangle in closest neigbors
            //define basis and project all points on it to fill HT
            for(int tp2=0;tp2<idNeigbors.size();tp2++)
            {
                //get index of point 2 in mVerticesDes
                int p2=idNeigbors[tp2];
                //define first basis vector
                Point2f basis1= mProjs[p2]-mProjs[p];
                
                for(int tp3=0;tp3<idNeigbors.size();tp3++)
                    if(p2!=idNeigbors[tp3])
                    {
                        int p3=idNeigbors[tp3];
                        //define second basis
                        Point2f basis2= mProjs[p3]-mProjs[p];
                        
                        //check direction of triangle
                        if(testDirectionBasis(basis1,basis2))
                        {
                            //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                            Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                            Matx22f tBasisInv=tBasis.inv();
                            
                            //good basis => project all points and fill HT
                            for(int i=0;i<mProjs.size();i++)
                                if(i!=p && i!=p2 && i!=p3)
                                {
                                    //project in current basis
                                    Point2f relCoord=tBasisInv*(mProjs[i]-mProjs[p]);
                                    
                                    //get bin
                                    Point2i bin=toCell(relCoord);
                                    
                                    //update HT with vite for p
                                    if(bin.x>=0 && bin.x<nbBinPerDim.x && bin.y>=0 && bin.y<nbBinPerDim.y)
                                        HashTable[bin.x*(nbBinPerDim.y*nbIds) + bin.y*nbIds + p]++;
                                }
                            
                        }
                    }
                
            }
        }
    }
    
    //probably would benefit from HT smoothing...
    //=> not any more as we do that with perspective transformation knowledge
    //actually could estimate variance of computed coordinates in local basis and blur using computed variance
    //blurHashTable();
}


void GH::blurHashTable()
{
    int radiusBlur=5;
    //create buffer
    int *HashTableBuff= new int[nbBinPerDim.x*nbBinPerDim.y];
    
    for(int id=0;id<nbIds;id++)
    {
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.x;j++)
            {
                float res=0;
                for(int i2=-radiusBlur;i2<=radiusBlur;i2++)
                    for(int j2=-radiusBlur;j2<=radiusBlur;j2++)
                    {
                        int ic=i+i2;
                        int jc=j+j2;
                        if(ic>=0 && ic<nbBinPerDim.x && jc>=0 && jc<nbBinPerDim.y)
                        {
                            //just set pyramidal coef eg for radius = 2 => [1 2 3 2 1]
                            int coef = (1+radiusBlur-abs(i2))*(1+radiusBlur-abs(j2));
                            res+=coef * HashTable[ic*(nbBinPerDim.y*nbIds) + jc*nbIds + id];
                        }
                    }
                HashTableBuff[i*nbBinPerDim.y + j]=res;
                        
            }
        
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.x;j++)
                HashTable[i*(nbBinPerDim.y*nbIds) + j*nbIds + id]=HashTableBuff[i*nbBinPerDim.y + j];
    }
    
    delete[] HashTableBuff;
}

struct findIdInDetections
{
    int id;
    findIdInDetections(int i) : id(i) {}
    bool operator () ( const DetectionGH& l) const
    {
        return id == l.id;
    }
};

//extract blobs, get there 3D position, check which point they correspond to in HashTable
void GH::getModelPointsFromImage(const cv::Mat& img, std::vector<DetectionGH> &matches) const
{
    //get blobs
    vector<KeyPoint> blobs;
    extractBlobs(img, blobs);
    
    //plot them
    //for(int p=0;p<blobs.size();p++)
    //    cv::circle(img, blobs[p].pt, (blobs[p].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
    
    //get list of points from blob (will have to be removed later as just a copy of blobs)
    vector<Point2f> mPoints;
    for(int p=0;p<blobs.size();p++)mPoints.push_back(toMeters(cameraCalibration.cameraMatrix,blobs[p].pt));
    
    //empty output vectors
    matches.clear();
    
    //loop through all points
    for(int p=0;p<mPoints.size();p++)
    {
        //for each point need to accumulate votes from HT
        float votesId[nbIds];
        //init all votes to 0
        for(int id=0;id<nbIds;id++)
            votesId[id]=0;
        
        //for each point have to find the nbPtBasis closest points
        vector<int> idNeigbors;
        getClosestNeigbors(p, mPoints, idNeigbors);
        
        //for each positively oriented possible triangle in closest neigbors
        //define basis and project all points on it to fill HT
        for(int tp2=0;tp2<idNeigbors.size();tp2++)
        {
            //get index of point 2 in mVerticesDes
            int p2=idNeigbors[tp2];
            
            //define first basis vector
            Point2f basis1= mPoints[p2]-mPoints[p];
            
            for(int tp3=0;tp3<idNeigbors.size();tp3++)
                if(p2!=idNeigbors[tp3])
                {
                    int p3=idNeigbors[tp3];
                    //define second basis
                    Point2f basis2= mPoints[p3]-mPoints[p];
                    
                    //check direction of triangle
                    if(testDirectionBasis(basis1,basis2))
                    {
                        //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                        Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                        Matx22f tBasisInv=tBasis.inv();
                        
                        //good basis => project all points and fill HT
                        for(int i=0;i<mPoints.size();i++)
                            if(i!=p && i!=p2 && i!=p3)
                            {
                                //project in current basis
                                Point2f relCoord=tBasisInv*(mPoints[i]-mPoints[p]);
                                
                                //get bin
                                Point2i bin=toCell(relCoord);
                                
                                //read HT with vote for p
                                if(bin.x>=0 && bin.x<nbBinPerDim.x && bin.y>=0 && bin.y<nbBinPerDim.y)
                                    for(int id=0;id<nbIds;id++)
                                        votesId[id]+=HashTable[bin.x*(nbBinPerDim.y*nbIds) + bin.y*nbIds + id];
                                
                            }
                        
                    }
                }
            
        }
        
        //if had votes, then find the id with max value
        int idPointEstim=-1;
        int nbVotesForId=0;
        for(int id=0;id<nbIds;id++)
            if(votesId[id]>nbVotesForId)
            {
                idPointEstim=id;
                nbVotesForId=votesId[id];
            }
        
        //find second best id to compute discriminative power
        int idPointSecondBest=-1;
        int nbVotesForSecondBest=0;
        for(int id=0;id<nbIds;id++)
            if(id!=idPointEstim && votesId[id]>nbVotesForSecondBest)
            {
                idPointSecondBest=id;
                nbVotesForSecondBest=votesId[id];
            }
        
        
        //add the point&id pair to output if id not already in list; if it is then need to check which one has most votes
        if(nbVotesForId>0)
        {
            std::vector<DetectionGH>::iterator it;
            it = find_if (matches.begin(), matches.end(), findIdInDetections(idPointEstim));
            
            if (it != matches.end())//point exist, check which one is the best
            {
                int posInList=it-matches.begin();
                if(matches[posInList].nbVotes<nbVotesForId)//if new one better than existing one, then replace it
                {
                    matches[posInList].position=blobs[p].pt;
                    matches[posInList].id=idPointEstim;
                    matches[posInList].nbVotes=nbVotesForId;
                    matches[posInList].discriminativePower=nbVotesForId-nbVotesForSecondBest;
                }
            }
            else
            {
                DetectionGH newMatch(blobs[p].pt,idPointEstim,nbVotesForId,nbVotesForId-nbVotesForSecondBest);
                matches.push_back(newMatch);
            }
        }
    }
}

void GH::saveToFile(const std::string& filename) const
{
    ofstream of(filename, ios::out | ios::binary);
    
    if (of.is_open())
    {
        of.write((char *)&nbIds, sizeof(int));
        of.write((char *)&nbBinPerDim.x, sizeof(int));  of.write((char *)&nbBinPerDim.y, sizeof(int));
        of.write((char *)&poseRelMin.x, sizeof(float)); of.write((char *)&poseRelMin.y, sizeof(float));
        of.write((char *)&poseRelMax.x, sizeof(float)); of.write((char *)&poseRelMax.y, sizeof(float));
        
        for(int id=0;id<nbIds;id++)
            for(int i=0;i<nbBinPerDim.x;i++)
                for(int j=0;j<nbBinPerDim.y;j++)
                    of.write((char *)&HashTable[i*(nbBinPerDim.y*nbIds) + j*nbIds + id], sizeof(int));
    }
    of.close();
}
void GH::loadFromFile(const std::string& filename)
{
    ifstream of(filename, ios::in | ios::binary);
    
    if (of.is_open())
    {
        of.read((char *)&nbIds, sizeof(int));
        of.read((char *)&nbBinPerDim.x, sizeof(int));  of.read((char *)&nbBinPerDim.y, sizeof(int));
        of.read((char *)&poseRelMin.x, sizeof(float)); of.read((char *)&poseRelMin.y, sizeof(float));
        of.read((char *)&poseRelMax.x, sizeof(float)); of.read((char *)&poseRelMax.y, sizeof(float));
        
        // TODO: What happens if HashTable already exists
        HashTable= new int[nbBinPerDim.x*nbBinPerDim.y*nbIds];
        
        for(int id=0;id<nbIds;id++)
            for(int i=0;i<nbBinPerDim.x;i++)
                for(int j=0;j<nbBinPerDim.y;j++)
                    of.read((char *)&HashTable[i*(nbBinPerDim.y*nbIds) + j*nbIds + id], sizeof(int));
    }
    else
    {
        std::cerr << "Could not open " << filename << std::endl;
        throw std::runtime_error("GH::loadFromFile > File not found!");
    }
    of.close();
}

void GH::extractBlobs(const cv::Mat& input, vector<KeyPoint> &blobs) const
{
    Mat gray;
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    
    // blob detector
    sbd->detect(gray, blobs);
}

//set model points
/*void GH::getSignatureBasis(vector<Point3f> &mVerticesDes, vector<int> &basisId, char *filename)
{
    int p=basisId[0];
    int p2=basisId[1];
    int p3=basisId[2];
    
    //create temp HashTable
    int HashTableTemp[nbBinPerDim.x*nbBinPerDim.y];
    for(int i=0;i<nbBinPerDim.x*nbBinPerDim.y;i++)HashTableTemp[i]=0;
    
    //define first basis vector
    Point3f basis1= mVerticesDes[p2]-mVerticesDes[p];
    Point3f basis2= mVerticesDes[p3]-mVerticesDes[p];
    
    //get third perpendicular base
    Point3f basis3=basis1.cross(basis2);
    
    //put three basis in a 3x3 matrix to inverse
    Matx33f tBasis(basis1.x,basis2.x,basis3.x,basis1.y,basis2.y,basis3.y,basis1.z,basis2.z,basis3.z);
    
    //compute inverse matrix to express all other points i this basis
    Matx33f tBasisInv=tBasis.inv();
    Point3f ibasis1(tBasisInv(0,0),tBasisInv(0,1),tBasisInv(0,2));
    Point3f ibasis2(tBasisInv(1,0),tBasisInv(1,1),tBasisInv(1,2));
    
    //good basis => project all points and fill HT
    for(int i=0;i<mVerticesDes.size();i++)
        if(i!=p && i!=p2 && i!=p3)
        {
            //project in current basis
            Point2f relCoord;
            //relCoord.x=(mVerticesDes[i]-mVerticesDes[p]).dot(ibasis1);
            //relCoord.y=(mVerticesDes[i]-mVerticesDes[p]).dot(ibasis2);
            //float normMax1=(norm(mVerticesDes[i]-mVerticesDes[p])>norm(basis1))?norm(mVerticesDes[i]-mVerticesDes[p]):norm(basis1);
            //float normMax2=(norm(mVerticesDes[i]-mVerticesDes[p])>norm(basis2))?norm(mVerticesDes[i]-mVerticesDes[p]):norm(basis2);
            float normMax1=norm(basis1);
            float normMax2=norm(basis2);
            relCoord.x=(mVerticesDes[i]-mVerticesDes[p]).dot(basis1)/(normMax1*normMax1);
            relCoord.y=(mVerticesDes[i]-mVerticesDes[p]).dot(basis2)/(normMax2*normMax2);
            
            //get bin
            Point2i bin=toCell(relCoord);
            
            cout<<relCoord<<"\t bin = "<<bin<<endl;
            
            //update HT with vite for p
            if(bin.x>=0 && bin.x<nbBinPerDim.x && bin.y>=0 && bin.y<nbBinPerDim.y)
                HashTableTemp[bin.x*nbBinPerDim.y + bin.y]++;
        }
    
    
    ofstream of;
    of.open (filename);
    
    if (of.is_open())
    {
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.y;j++)
                of<<i<<"\t"<<j<<"\t"<<HashTableTemp[i*nbBinPerDim.y + j]<<endl;
        of.close();
    }
}*/
/*
void GH::convertToWorldFrame(vector<KeyPoint> &keypoint,Mat &cameraMatrix, Mat &distCoeffs,vector<Point3f> &points3d)
{
    double fx=cameraMatrix.at<double>(0,0);    double fy=cameraMatrix.at<double>(1,1);
    double cx=cameraMatrix.at<double>(0,2);    double cy=cameraMatrix.at<double>(1,2);
    for(int i=0;i<keypoint.size();i++)
    {
        Point3f newPoint;
        //use linear regression with scale and inertia to estimate inverse depth
        //float inverseDepthReg=0.00130515*keypoint[i].size -0.01756429*keypoint[i].response  + 0.01517956;
        float inverseDepthReg=keypoint[i].size;
        newPoint.x=(keypoint[i].pt.x-cx)/fx/inverseDepthReg;
        newPoint.y=(keypoint[i].pt.y-cy)/fy/inverseDepthReg;
        newPoint.z=1./inverseDepthReg;
        //cout<<"Size = "<<keypoint[i].size<<"\tInertia = "<<keypoint[i].response<<"\tInvDepth = "<<inverseDepthReg<<endl;
        
        //if want to just use 2D info without depth, comment previous lines and use:
        float zDefault=20.;
        newPoint.x=zDefault*(keypoint[i].pt.x-cx)/fx;
        newPoint.y=zDefault*(keypoint[i].pt.y-cy)/fy;
        newPoint.z=zDefault;
    
        points3d.push_back(newPoint);
    }
    
    //output 3D coordinates to file for displaying
    ofstream of;
    of.open ("/Users/amaurydame/Projects/MBTracker/files/BackProject.dat");
    
    if (of.is_open())
    {
        for(int i=0;i<points3d.size();i++)
             of<<points3d[i].x<<"\t"<<points3d[i].y<<"\t"<<points3d[i].z<<endl;
        of.close();
    }
    
}
*/

}
