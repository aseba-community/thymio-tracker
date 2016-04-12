
#include "GHscale.hpp"

#include <stdexcept>

#include <opencv2/calib3d.hpp> //solvePnP
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

namespace thymio_tracker
{

GHscale::GHscale(IntrinsicCalibration *_camCalib)
{
    cameraCalibration_ptr=_camCalib;
    //for now we will consider 3 neigboring points to define bases
    nbPtBasis=2;
    
    //pablo s settings
    cv::SimpleBlobDetector::Params params;
    params.thresholdStep = 10;
    params.minThreshold = 80;
    params.maxThreshold = 200;
    params.minDistBetweenBlobs = 4;
    params.minRepeatability = 3;

    params.filterByColor = true;
    params.blobColor = 0;

    params.filterByArea = true;
    params.minArea = 5;
    params.maxArea = 800;

    params.filterByCircularity = true;
    params.minCircularity = 0.6;
    params.maxCircularity = 1.4;

    params.filterByInertia = false;
    params.minInertiaRatio = 0.7;
    params.maxInertiaRatio = 1.3;

    params.filterByConvexity = false;
    sbd = cv::SimpleBlobDetector::create(params);
    
    /*cv::SimpleBlobDetector::Params params;
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
    sbd = cv::SimpleBlobDetector::create(params);*/
}

GHscale::~GHscale()
{
    delete[] HashTable;
}

//set Hashing table, could do that autonomously from model, but as we have one object we code it the hard way
void GHscale::initHashTable(int _nbIds, Point3i _nbBinsPerDim)
{
    nbIds=_nbIds;
    nbBinPerDim=_nbBinsPerDim;
    HashTable= new float[nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z*nbIds];
    
    //init all bins to 0 votes
    for(int i=0;i<nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z*nbIds;i++)
        HashTable[i]=0;
}

Point3f GHscale::toCell(const Point3f& relativePos) const
{
    Point3f res;
    res.x=(float)nbBinPerDim.x*(relativePos.x-poseRelMin.x)/(poseRelMax.x-poseRelMin.x);
    res.y=(float)nbBinPerDim.y*(relativePos.y-poseRelMin.y)/(poseRelMax.y-poseRelMin.y);
    res.z=(float)nbBinPerDim.z*(relativePos.z-poseRelMin.z)/(poseRelMax.z-poseRelMin.z);
    return res;
}

struct sort_wrt_second {
    bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
        return left.second < right.second;
    }
};


void GHscale::getClosestNeigbors(unsigned int p, const vector<Point3f>& mVerticesDes, vector<unsigned int>& idNeigbors) const
{
    //create pairs of point indexes and corresponding distance and sort with respect to deistance
    vector< pair<int,float> > pairIdDist;
    for(unsigned int i=0;i<mVerticesDes.size();i++)
        if(i!=p)
    {
        pair<int,float> newPair;
        newPair.first=i;
        newPair.second=norm(Pointxy(mVerticesDes[i]-mVerticesDes[p]));
        pairIdDist.push_back(newPair);
    }
    
    //sort it
    std::sort(pairIdDist.begin(), pairIdDist.end(), sort_wrt_second());
    
    //return first elements
    for(unsigned int i=0;i<pairIdDist.size() && i<nbPtBasis;i++)
        idNeigbors.push_back(pairIdDist[i].first);
    
}

void GHscale::addVoteToBin(const cv::Point3f& bin,const int &id, const float _v)
{
    //nearest neigbor
    /*if(bin.x>=0 && bin.x<nbBinPerDim.x &&
       bin.y>=0 && bin.y<nbBinPerDim.y &&
       bin.z>=0 && bin.z<nbBinPerDim.z)
        HashTable[(int)bin.x*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (int)bin.y*nbBinPerDim.z*nbIds + (int)bin.z*nbIds + id]+=_v;
    */
    //Simple spline
    if(bin.x>=0 && bin.x<nbBinPerDim.x &&
       bin.y>=0 && bin.y<nbBinPerDim.y &&
       bin.z>=0 && bin.z<nbBinPerDim.z)
    {
        int Ex=(int)bin.x;float ex=bin.x-Ex;
        int Ey=(int)bin.y;float ey=bin.y-Ey;
        int Ez=(int)bin.z;float ez=bin.z-Ez;
        HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + Ez*nbIds + id]+=(1.-ex)*(1.-ey)*(1.-ez)*_v;
        HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id]+=(1.-ex)*(1.-ey)*(ez)*_v;
        HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + Ez*nbIds + id]+=(1.-ex)*(ey)*(1.-ez)*_v;
        HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id]+=(1.-ex)*(ey)*(ez)*_v;
        
        HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + Ez*nbIds + id]+=(ex)*(1.-ey)*(1.-ez)*_v;
        HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id]+=(ex)*(1.-ey)*(ez)*_v;
        HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + Ez*nbIds + id]+=(ex)*(ey)*(1.-ez)*_v;
        HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id]+=(ex)*(ey)*(ez)*_v;
    }
    
}

void GHscale::readVotesFromBin(const cv::Point3f& bin,float *votes) const
{
    //nearest neigbor
    /*if(bin.x>=0 && bin.x<nbBinPerDim.x && bin.y>=0 && bin.y<nbBinPerDim.y && bin.z>=0 && bin.z<nbBinPerDim.z)
        for(int id=0;id<nbIds;id++)
            votes[id]+=HashTable[(int)bin.x*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (int)bin.y*nbBinPerDim.z*nbIds + (int)bin.z*nbIds + id];*/
    
    if(bin.x>=0 && bin.x<nbBinPerDim.x && bin.y>=0 && bin.y<nbBinPerDim.y && bin.z>=0 && bin.z<nbBinPerDim.z)
    {
        int Ex=(int)bin.x;float ex=bin.x-Ex;
        int Ey=(int)bin.y;float ey=bin.y-Ey;
        int Ez=(int)bin.z;float ez=bin.z-Ez;
        for(int id=0;id<nbIds;id++)
        {
            votes[id]+=(1.-ex)*(1.-ey)*(1.-ez)*HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + Ez*nbIds + id];
            votes[id]+=(1.-ex)*(1.-ey)*(ez)*HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id];
            votes[id]+=(1.-ex)*(ey)*(1.-ez)*HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + Ez*nbIds + id];
            votes[id]+=(1.-ex)*(ey)*(ez)*HashTable[Ex*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id];
            
            votes[id]+=(ex)*(1.-ey)*(1.-ez)*HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + Ez*nbIds + id];
            votes[id]+=(ex)*(1.-ey)*(ez)*HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + Ey*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id];
            votes[id]+=(ex)*(ey)*(1.-ez)*HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + Ez*nbIds + id];
            votes[id]+=(ex)*(ey)*(ez)*HashTable[(1+Ex)*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + (1+Ey)*nbBinPerDim.z*nbIds + (1+Ez)*nbIds + id];
            
        }
    }
}

void GHscale::setModel(vector<Point3f> *projPoints, int nbPoses)
{
    //get hash table limits
    //initialise with default mean value (0,0) coordinates and relScale of 1
    poseRelMin = Point3f(0,0,1); poseRelMax = Point3f(0,0,1);
    //get all the bases and project all point and set limits accordingly to support
    for(int idpose=0;idpose<nbPoses;idpose++)
    {
        vector<Point3f> &mProjs=projPoints[idpose];
        //loop through all points
        for(unsigned int p=0;p<mProjs.size();p++)
        {
            //for each point have to find the nbPtBasis closest points
            vector<unsigned int> idNeigbors;
            getClosestNeigbors(p,mProjs,idNeigbors);
            
            
            //for each positively oriented possible triangle in closest neigbors
            //define basis and project all points on it to fill HT
            for(unsigned int tp2=0;tp2<idNeigbors.size();tp2++)
            {
                //get index of point 2 in mVerticesDes
                unsigned int p2=idNeigbors[tp2];
                //define first basis vector
                Point2f basis1= Pointxy(mProjs[p2]-mProjs[p]);
                
                for(unsigned int tp3=0;tp3<idNeigbors.size();tp3++)
                    if(p2!=idNeigbors[tp3])
                    {
                        unsigned int p3=idNeigbors[tp3];
                        //define second basis
                        Point2f basis2= Pointxy(mProjs[p3]-mProjs[p]);
                        
                        //check direction of triangle
                        if(testDirectionBasis(basis1,basis2))
                        {
                            //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                            Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                            Matx22f tBasisInv=tBasis.inv();
                            
                            //good basis => project all points and fill HT
                            for(unsigned int i=0;i<mProjs.size();i++)
                                if(i!=p && i!=p2 && i!=p3)
                                {
                                    //project in current basis
                                    Point2f relCoord=tBasisInv*(Pointxy(mProjs[i]-mProjs[p]));
                                    float relScale=mProjs[i].z/mProjs[p].z;
                                    
                                    //readjust limits
                                    if(relCoord.x<poseRelMin.x)poseRelMin.x=relCoord.x;
                                    if(relCoord.y<poseRelMin.y)poseRelMin.y=relCoord.y;
                                    if(relScale<poseRelMin.z)poseRelMin.z=relScale;
                                    if(relCoord.x>poseRelMax.x)poseRelMax.x=relCoord.x;
                                    if(relCoord.y>poseRelMax.y)poseRelMax.y=relCoord.y;
                                    if(relScale>poseRelMax.z)poseRelMax.z=relScale;
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
    float diffRellz=poseRelMax.z-poseRelMin.z;
    poseRelMin.x=poseRelMin.x-marginRel*diffRellx;
    poseRelMin.y=poseRelMin.y-marginRel*diffRelly;
    poseRelMin.z=poseRelMin.z-marginRel*diffRellz;
    poseRelMax.x=poseRelMax.x+marginRel*diffRellx;
    poseRelMax.y=poseRelMax.y+marginRel*diffRelly;
    poseRelMax.z=poseRelMax.z+marginRel*diffRellz;

    //now that we have margin, fill tables with votes
    for(int idpose=0;idpose<nbPoses;idpose++)
    {
        vector<Point3f> &mProjs = projPoints[idpose];
    
        //loop through all points to fill HT
        for(unsigned int p=0;p<mProjs.size();p++)
        {
            //for each point have to find the nbPtBasis closest points
            vector<unsigned int> idNeigbors;
            getClosestNeigbors(p,mProjs,idNeigbors);
            
            
            //for each positively oriented possible triangle in closest neigbors
            //define basis and project all points on it to fill HT
            for(unsigned int tp2=0;tp2<idNeigbors.size();tp2++)
            {
                //get index of point 2 in mVerticesDes
                unsigned int p2=idNeigbors[tp2];
                //define first basis vector
                Point2f basis1= Pointxy(mProjs[p2]-mProjs[p]);
                
                for(unsigned int tp3=0;tp3<idNeigbors.size();tp3++)
                    if(p2!=idNeigbors[tp3])
                    {
                        unsigned int p3=idNeigbors[tp3];
                        //define second basis
                        Point2f basis2= Pointxy(mProjs[p3]-mProjs[p]);
                        
                        //check direction of triangle
                        if(testDirectionBasis(basis1,basis2))
                        {
                            //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                            Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                            Matx22f tBasisInv=tBasis.inv();
                            
                            //good basis => project all points and fill HT
                            for(unsigned int i=0;i<mProjs.size();i++)
                                if(i!=p && i!=p2 && i!=p3)
                                {
                                    //project in current basis
                                    Point2f relCoord=tBasisInv*(Pointxy(mProjs[i]-mProjs[p]));
                                    float relScale=mProjs[i].z/mProjs[p].z;
                                    Point3f relFull(relCoord.x,relCoord.y,relScale);
                                    
                                    //get bin
                                    Point3f bin=toCell(relFull);
                                   
                                    //update HT with vite for p
                                    addVoteToBin(bin,p,1.);
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


void GHscale::blurHashTable()
{
    //blur on x.y dimension
    int radiusBlur=5;
    //create buffer
    float *HashTablexyBuff= new float[nbBinPerDim.x*nbBinPerDim.y];
    
    for(int id=0;id<nbIds;id++)
        for(int k=0;k<nbBinPerDim.z;k++)
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
                            res+=coef * HashTable[ic*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + jc*nbBinPerDim.z*nbIds + k*nbIds + id];
                        }
                    }
                HashTablexyBuff[i*nbBinPerDim.y + j]=res;
                        
            }
        
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.x;j++)
                HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id]=HashTablexyBuff[i*nbBinPerDim.y + j];
    }
    
    delete[] HashTablexyBuff;
    
    //blur the scale dimension
    float *HashTablescaleBuff= new float[nbBinPerDim.z];
    for(int i=0;i<nbBinPerDim.x;i++)
        for(int j=0;j<nbBinPerDim.x;j++)
            for(int id=0;id<nbIds;id++)
            {
                for(int k=0;k<nbBinPerDim.z;k++)
                {
                    float res=0;
                    for(int k2=-radiusBlur;k2<=radiusBlur;k2++)
                    {
                        int kc=k+k2;
                        if(kc>=0 && kc<nbBinPerDim.z)
                        {
                            //just set pyramidal coef eg for radius = 2 => [1 2 3 2 1]
                            int coef = (1+radiusBlur-abs(k2))*(1+radiusBlur-abs(k2));
                            res+=coef * HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + kc*nbIds + id];
                        }
                        
                    }
                    HashTablescaleBuff[k]=res;
                }
                
                for(int k=0;k<nbBinPerDim.z;k++)
                    HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id]=HashTablescaleBuff[k];
            }
    delete[] HashTablescaleBuff;
    
    //set all value along z equal
    /*for(int i=0;i<nbBinPerDim.x;i++)
        for(int j=0;j<nbBinPerDim.x;j++)
            for(int id=0;id<nbIds;id++)
            {
                float res=0;
                for(int k=0;k<nbBinPerDim.z;k++)
                    res+=HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id];
                
                for(int k=0;k<nbBinPerDim.z;k++)
                    HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id]=res;
            }*/
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
void GHscale::getModelPointsFromImage(const cv::Mat& img, std::vector<DetectionGH> &matches) const
{
    //get blobs
    vector<KeyPoint> blobs;
    extractBlobs(img, blobs);
    
    //plot them
    for(unsigned int p=0;p<blobs.size();p++)
        cv::circle(img, blobs[p].pt, (blobs[p].size - 1) / 2 + 1, cv::Scalar(255, 0, 0), -1);
    
    getModelPointsFromImage(blobs,matches);
}

void GHscale::getModelPointsFromImage(const vector<KeyPoint> &blobs, std::vector<DetectionGH> &matches) const
{
    //get list of points from blob (will have to be removed later as just a copy of blobs)
    vector<Point3f> mPoints;
    for(unsigned int p=0;p<blobs.size();p++)
    {
        Point2f m = toMeters(cameraCalibration_ptr->cameraMatrix,blobs[p].pt);
        mPoints.push_back(Point3f(m.x,m.y,blobs[p].size));
    }
    
    //empty output vectors
    matches.clear();
    
    //loop through all points
    for(unsigned int p=0;p<mPoints.size();p++)
    {
        //for each point need to accumulate votes from HT
        float votesId[nbIds];
        //init all votes to 0
        for(int id=0;id<nbIds;id++)
            votesId[id]=0;
        
        //for each point have to find the nbPtBasis closest points
        vector<unsigned int> idNeigbors;
        getClosestNeigbors(p, mPoints, idNeigbors);
        
        //for each positively oriented possible triangle in closest neigbors
        //define basis and project all points on it to fill HT
        for(unsigned int tp2=0;tp2<idNeigbors.size();tp2++)
        {
            //get index of point 2 in mVerticesDes
            unsigned int p2=idNeigbors[tp2];
            
            //define first basis vector
            Point2f basis1= Pointxy(mPoints[p2]-mPoints[p]);
            
            for(unsigned int tp3=0;tp3<idNeigbors.size();tp3++)
                if(p2!=idNeigbors[tp3])
                {
                    unsigned int p3=idNeigbors[tp3];
                    //define second basis
                    Point2f basis2= Pointxy(mPoints[p3]-mPoints[p]);
                    
                    //check direction of triangle
                    if(testDirectionBasis(basis1,basis2))
                    {
                        //put basis in a 2x2 matrix to inverse it and express all other points i this basis
                        Matx22f tBasis(basis1.x,basis2.x,basis1.y,basis2.y);
                        Matx22f tBasisInv=tBasis.inv();
                        
                        //good basis => project all points and fill HT
                        for(unsigned int i=0;i<mPoints.size();i++)
                            if(i!=p && i!=p2 && i!=p3)
                            {
                                //project in current basis
                                Point2f relCoord=tBasisInv*(Pointxy(mPoints[i]-mPoints[p]));
                                float relScale=mPoints[i].z/mPoints[p].z;
                                Point3f relFull(relCoord.x,relCoord.y,relScale);
                                
                                //get bin
                                Point3f bin=toCell(relFull);
                                
                                //read HT with vote for p
                                readVotesFromBin(bin, &votesId[0]);
                            }
                        
                    }
                }
            
        }
        
        //if had votes, then find the id with max value
        int idPointEstim=-1;
        float nbVotesForId=0;
        for(int id=0;id<nbIds;id++)
            if(votesId[id]>nbVotesForId)
            {
                idPointEstim=id;
                nbVotesForId=votesId[id];
            }
        
        //find second best id to compute discriminative power
        int idPointSecondBest=-1;
        (void)idPointSecondBest;
        float nbVotesForSecondBest=0;
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

void GHscale::saveToStream(std::ostream& os) const
{
    os.write((char *)&nbIds, sizeof(int));
    os.write((char *)&nbBinPerDim.x, sizeof(int));
    os.write((char *)&nbBinPerDim.y, sizeof(int));
    os.write((char *)&nbBinPerDim.z, sizeof(int));
    os.write((char *)&poseRelMin.x, sizeof(float)); os.write((char *)&poseRelMin.y, sizeof(float));
    os.write((char *)&poseRelMax.x, sizeof(float)); os.write((char *)&poseRelMax.y, sizeof(float));
    os.write((char *)&poseRelMin.z, sizeof(float)); os.write((char *)&poseRelMax.z, sizeof(float));
    
    for(int id=0;id<nbIds;id++)
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.y;j++)
                for(int k=0;k<nbBinPerDim.z;k++)
                os.write((char *)&HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id], sizeof(float));

}

void GHscale::saveToFileStorage(cv::FileStorage& fs) const
{

    //cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    cv::write(fs, "nbIds", nbIds);
    cv::write(fs, "nbBinPerDim", nbBinPerDim);
    cv::write(fs, "poseRelMin", poseRelMin);
    cv::write(fs, "poseRelMax", poseRelMax);

    //convert array into Matrix
    cv::Mat HTmat = cv::Mat(1, nbIds*nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z, CV_32FC1, HashTable, 2);
    cv::write(fs, "HTmat",HTmat);

    fs.release();
}

void GHscale::loadFromFileStorage(cv::FileStorage& fs)
{

    //cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    cv::read(fs["nbIds"], nbIds,0);
    cv::read(fs["nbBinPerDim"], nbBinPerDim,cv::Point3i());
    cv::read(fs["poseRelMin"], poseRelMin,cv::Point3f());
    cv::read(fs["poseRelMax"], poseRelMax,cv::Point3f());

    //convert array into Matrix
    cv::Mat HTmat;
    cv::read(fs["HTmat"],HTmat);

    HashTable = new float[nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z*nbIds];
    float *buff = (float*)(HTmat.data);
    for(int i=0;i<nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z*nbIds;i++)
        HashTable[i] = buff[i];

    fs.release();
}

void GHscale::loadFromStream(std::istream& is)
{
    is.read((char *)&nbIds, sizeof(int));
    is.read((char *)&nbBinPerDim.x, sizeof(int));
    is.read((char *)&nbBinPerDim.y, sizeof(int));
    is.read((char *)&nbBinPerDim.z, sizeof(int));
    is.read((char *)&poseRelMin.x, sizeof(float)); is.read((char *)&poseRelMin.y, sizeof(float));
    is.read((char *)&poseRelMax.x, sizeof(float)); is.read((char *)&poseRelMax.y, sizeof(float));
    is.read((char *)&poseRelMin.z, sizeof(float)); is.read((char *)&poseRelMax.z, sizeof(float));
    
    // TODO: What happens if HashTable already exists
    HashTable= new float[nbBinPerDim.x*nbBinPerDim.y*nbBinPerDim.z*nbIds];
    
    for(int id=0;id<nbIds;id++)
        for(int i=0;i<nbBinPerDim.x;i++)
            for(int j=0;j<nbBinPerDim.y;j++)
                for(int k=0;k<nbBinPerDim.z;k++)
                is.read((char *)&HashTable[i*(nbBinPerDim.y*nbBinPerDim.z*nbIds) + j*nbBinPerDim.z*nbIds + k*nbIds + id], sizeof(float));
}

void GHscale::extractBlobs(const cv::Mat& input, vector<KeyPoint> &blobs) const
{
    Mat gray;
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    
    // blob detector
    sbd->detect(gray, blobs);
}

//set model points
/*void GHscale::getSignatureBasis(vector<Point3f> &mVerticesDes, vector<int> &basisId, char *filename)
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
void GHscale::convertToWorldFrame(vector<KeyPoint> &keypoint,Mat &cameraMatrix, Mat &distCoeffs,vector<Point3f> &points3d)
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
