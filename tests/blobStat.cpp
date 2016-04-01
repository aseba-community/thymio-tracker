//track the robot s and blobs in one easy sequence so that we can get
//statistics on blobs, like how scale varies with depth, the variance
//of the scale, ...

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

//opencv display
Mat inputImage;
char window_name[] = "InputWindow";

//3D model, first simple version with just top
#include "GH.hpp"


//do tracking of blobs and computation of object pose to estimate relationship between scale blob and inertia with respect to depth
cv::Ptr<cv::SimpleBlobDetector> sbdm;
void extractBlobs(const cv::Mat& input, vector<KeyPoint> &blobs)
{
    
    if(sbdm.empty())
    {
        cv::SimpleBlobDetector::Params params;
        params.thresholdStep = 10;
        params.minThreshold = 40;
        params.maxThreshold = 200;
        params.minDistBetweenBlobs = 2;
        params.minRepeatability = 3;
        
        params.filterByColor = true;
        params.blobColor = 0;
        
        params.filterByArea = true;
        params.minArea = 1;
        params.maxArea = 50000;
        
        params.filterByCircularity = true;
        params.minCircularity = 0.8;
        params.maxCircularity = 1.2;
        
        params.filterByInertia = true;
        params.minInertiaRatio = 0.2;
        params.maxInertiaRatio = 1.8;
        
        params.filterByConvexity = false;
        //sbdm = cv::SimpleBlobDetector::create(params);
        sbdm = cv::SimpleBlobDetector::create(params);
    }
    Mat gray;
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    
    // blob detector
    sbdm->detect(gray, blobs);
}

int TrackBlobsFindPoseAndGetStats()
{
    //load intrinsic parameters
    FileStorage fs;    fs.open("../Data/calibration/nexus_camera_calib.xml", FileStorage::READ);
    Mat cameraMatrix;    fs["camera_matrix"] >> cameraMatrix;
    Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    
    //init display
    namedWindow( window_name,WINDOW_AUTOSIZE);
    
    //init model
    vector<Point3f> mVerticesDes;
    mVerticesDes.push_back(Point3f(0,5.6,0));
    mVerticesDes.push_back(Point3f(0,4.8,0));
    mVerticesDes.push_back(Point3f(0,0.8,0));
    mVerticesDes.push_back(Point3f(0,0,0));
    
    mVerticesDes.push_back(Point3f(0.8,4.8,0));
    mVerticesDes.push_back(Point3f(0.8,0.8,0));
    mVerticesDes.push_back(Point3f(0.8,0,0));
    
    mVerticesDes.push_back(Point3f(8.,4.8,0));
    mVerticesDes.push_back(Point3f(8.,0.8,0));
    mVerticesDes.push_back(Point3f(8.,0,0));
    
    mVerticesDes.push_back(Point3f(8.8,5.6,0));
    mVerticesDes.push_back(Point3f(8.8,4.8,0));
    mVerticesDes.push_back(Point3f(8.8,0.8,0));
    mVerticesDes.push_back(Point3f(8.8,0,0));
    
    //Init corresponding blob position in first image
    vector<Point2f> mVertProjDes;
    mVertProjDes.push_back(Point2f(1345,836));
    mVertProjDes.push_back(Point2f(1345,750));
    mVertProjDes.push_back(Point2f(1343,339));
    mVertProjDes.push_back(Point2f(1342,260));
    
    mVertProjDes.push_back(Point2f(1263,743));
    mVertProjDes.push_back(Point2f(1264,343));
    mVertProjDes.push_back(Point2f(1266,262));
    
    mVertProjDes.push_back(Point2f(532,734));
    mVertProjDes.push_back(Point2f(538,330));
    mVertProjDes.push_back(Point2f(543,250));
    
    mVertProjDes.push_back(Point2f(442,817));
    mVertProjDes.push_back(Point2f(449,737));
    mVertProjDes.push_back(Point2f(466,328));
    mVertProjDes.push_back(Point2f(467,249));
    
    //for depth regression from scale and inertia
    //ofstream of;
    //of.open ("..Data/DepthRegData.dat");

    
    //object pose
    cv::Mat rvec,tvec;
    //stats on scale wrt inv depth
    float ratio_variance=0;
    float accumul_ratio=0;
    
    //stats with respect to pairs of close blobs
    //basically should have sqrt(scale1*scale2)/d12 constant
    float accu_d_on_ss=0;
    float norm_d_on_ss=0;
    for(int image=1;image<600;image++)
    {
        //read first image
        cout<<"Process image "<<image<<endl;
        char file_name[100];
        sprintf(file_name, "/Users/amaurydame/Data/nexus/TrackSeq3/image-%03d.png", image);
        inputImage = imread(file_name, 1 );
        
        //get blobs in current image
        vector<KeyPoint> mBlobs;
        extractBlobs(inputImage,mBlobs);
        
        if(mBlobs.size()==0)
        {
            cout<<"No blobs detected in first image"<<endl;
            return 0;
        }

        //display blobs
        for(unsigned int v=0;v<mBlobs.size();v++)
            circle(inputImage, mBlobs[v].pt, 4, Scalar(250,100,250), -1, 8, 0);
      
        
        //find closest blobs to mVertProjDes
        vector<KeyPoint> mBlobsNoRansac;
        vector<Point2f> mBlobsP2fNoRansac;
        vector<Point3f> m3DVerticesNoRansac;
        
        //for each projected model point mVertProjDes, search for closest blob
        for(unsigned int v=0;v<mVertProjDes.size();v++)
        {
            //init as first blob
            KeyPoint bestMatch=mBlobs[0];
            float distMin=norm(bestMatch.pt-mVertProjDes[v]);
            
            //search best match
            for(unsigned int i=1;i<mBlobs.size();i++)
            {
                float distCurr=norm(mBlobs[i].pt-mVertProjDes[v]);
                if(distCurr<distMin)
                {
                    distMin=distCurr;
                    bestMatch=mBlobs[i];
                }
            }
            
            //put it in list matches if good enough
            if(distMin<20)
            {
                mBlobsNoRansac.push_back(bestMatch);
                mBlobsP2fNoRansac.push_back(Point2f(bestMatch.pt));
                m3DVerticesNoRansac.push_back(mVerticesDes[v]);
            }
        }

        //compute pose from matches
        if(mBlobsP2fNoRansac.size()>=4)
        {
            cv::solvePnP(m3DVerticesNoRansac, mBlobsP2fNoRansac, cameraMatrix, distCoeffs, rvec, tvec);
            
            //project found Vetrices on image and check if reprojection error is small enough
            vector<Point2f> mProjectedVerticesFound;
            projectPoints(m3DVerticesNoRansac, rvec, tvec, cameraMatrix, distCoeffs, mProjectedVerticesFound);
            
            for(unsigned int v=0;v<mProjectedVerticesFound.size();v++)
            {
                float distCurr=norm(mProjectedVerticesFound[v]-mBlobsP2fNoRansac[v]);
                //remove them if bigger than threshold
                if(distCurr>5)
                {
                    mProjectedVerticesFound.erase(mProjectedVerticesFound.begin()+v);
                    mBlobsNoRansac.erase(mBlobsNoRansac.begin()+v);
                    mBlobsP2fNoRansac.erase(mBlobsP2fNoRansac.begin()+v);
                    m3DVerticesNoRansac.erase(m3DVerticesNoRansac.begin()+v);
                    v--;
                }
            }
            //recompute pose after removing possible outliers
            cv::solvePnP(m3DVerticesNoRansac, mBlobsP2fNoRansac, cameraMatrix, distCoeffs, rvec, tvec);
            
            //project again but all vertices from model so that can search for new position of blob from there at next frame
            vector<Point2f> mProjectedVertices;
            projectPoints(mVerticesDes, rvec, tvec, cameraMatrix, distCoeffs, mProjectedVertices);
            for(unsigned int v=0;v<mVerticesDes.size();v++)
                circle(inputImage, mProjectedVertices[v], 4, Scalar(250,250,100), -1, 8, 0);
            
            //replace desired pix position by projected ones
            for(unsigned int v=0;v<mVerticesDes.size();v++)mVertProjDes[v]=mProjectedVertices[v];
            
            //now can compute our stats
            //relative scale variation:
            Affine3d r_to_c(rvec,tvec);
            for(unsigned int v=0;v<mProjectedVerticesFound.size();v++)
                for(unsigned int v2=0;v2<mProjectedVerticesFound.size();v2++)
                    if(v!=v2)
                {
                    Point3f point1Cam=r_to_c*m3DVerticesNoRansac[v];
                    Point3f point2Cam=r_to_c*m3DVerticesNoRansac[v2];
                    //ratio of scales should be equal to ratio of inverted detph (GT)
                    float ratio_inv_depth=point2Cam.z/point1Cam.z;
                    float ratio_scale=mBlobsNoRansac[v].size/mBlobsNoRansac[v2].size;
                    //cout<<"ratio_inv_depth = "<<ratio_inv_depth<<endl;
                    //cout<<"ratio_scale = "<<ratio_scale<<endl;
                    
                    ratio_variance+=(ratio_inv_depth-ratio_scale)*(ratio_inv_depth-ratio_scale);
                    accumul_ratio++;

                }
            
            //get all pairs of close points from model and compute pair statistics
            for(unsigned int v=0;v<mProjectedVerticesFound.size();v++)
                for(unsigned int v2=0;v2<mProjectedVerticesFound.size();v2++)
                    if(v!=v2)
                    {
                        //check if 3d points are close enough to be in same cluster
                        if(norm(m3DVerticesNoRansac[v]-m3DVerticesNoRansac[v2])<0.9)
                        {
                            float d_on_ss=norm(mBlobsNoRansac[v].pt-mBlobsNoRansac[v2].pt)/sqrt(mBlobsNoRansac[v].size*mBlobsNoRansac[v2].size);
                            accu_d_on_ss+=d_on_ss;
                            norm_d_on_ss++;
                            
                            //cout<<"d_on_ss = "<<d_on_ss<<endl;
                            
                            cout<<"scale dist = "<<sqrt((mBlobsNoRansac[v].size-mBlobsNoRansac[v2].size)*(mBlobsNoRansac[v].size-mBlobsNoRansac[v2].size))<<std::endl;
                        }
                    }
            

        }
        
        //show image
        imshow(window_name,inputImage);
        waitKey(1);
    }

    cout<<"variance ratio scale = "<<sqrt(ratio_variance/accumul_ratio)<<endl;//end up with variance of 0.25
    cout<<"d_on_ss = "<<accu_d_on_ss/norm_d_on_ss<<endl;//end up with variance of 0.25
    //of.close();
    
    return 0;
}
int main( int argc, char** argv )
{
    cout<<"Find blob stats"<<endl;
    TrackBlobsFindPoseAndGetStats();
    return 0;

}