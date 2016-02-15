#include "Models.hpp"

using namespace cv;
using namespace std;

namespace thymio_tracker
{

Object3D::Object3D()
{
    lengthHistory = 4;
    nbHypoPerTime = 4;
    PoseHypothesisHistory = new PoseHypothesisSet*[lengthHistory];
    for(int i=0;i<nbHypoPerTime;i++)
        PoseHypothesisHistory[i] = new PoseHypothesisSet[nbHypoPerTime];
}
Object3D::~Object3D()
{
    for(int i=0;i<nbHypoPerTime;i++)
        delete[] PoseHypothesisHistory[i];
    delete[] PoseHypothesisHistory;
}

void Object3D::draw(Mat& img, const Mat& cameraMatrix, const Mat& distCoeffs, const Affine3d& poseCam) const
{
    for(unsigned int e=0;e<mEdges.size();e++)
        drawEdge(mEdges[e], img, cameraMatrix, distCoeffs, poseCam);
    
    for(unsigned int v=0;v<mVertices.size();v++)
        drawVertice(mVertices[v], img, cameraMatrix, distCoeffs, poseCam);
    
    //draw object frame (axis XYZ)
    vector<Point3f> framePoints;
    framePoints.push_back(Point3f(0,0,0));
    framePoints.push_back(Point3f(0.03,0,0));
    framePoints.push_back(Point3f(0,0.03,0));
    framePoints.push_back(Point3f(0,0,0.03));

    for(int p=0;p<4;p++)framePoints[p]=pose*framePoints[p];
    vector<Point2f> vprojVertices;
    projectPoints(framePoints, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    line(img, vprojVertices[0], vprojVertices[1], Scalar(0,0,255), 2);
    line(img, vprojVertices[0], vprojVertices[2], Scalar(0,255,0), 2);
    line(img, vprojVertices[0], vprojVertices[3], Scalar(255,0,0), 2);

}

vector<Point2f> Object3D::projectVertices(const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    vector<Point2f> vProjPoints;
    Affine3d poseComb=poseCam * pose;
    
    //project all points
    projectPoints(mVertices, poseComb.rvec(), poseComb.translation(), cameraMatrix, distCoeffs, vProjPoints);
    
    return vProjPoints;
}

void Object3D::drawVertice(const Point3f &_vertice, Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    //transform using object pose
    vector<Point3f> pobj; pobj.push_back(pose*_vertice);
    Point3f pcam; pcam= poseCam * pose * pobj[0];
    
    //project
    vector<Point2f> vprojVertices;
    projectPoints(pobj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    
    if(pcam.z>0 && 1./pcam.z > 0)
        circle(img, vprojVertices[0], 3./pcam.z, Scalar(255,255,0),2);
}

void Object3D::drawEdge(const ModelEdge &_edge, const Mat &img, const Mat &cameraMatrix, const Mat &distCoeffs, const Affine3d &poseCam) const
{
    vector<Point3f> LineObj; LineObj.push_back(pose*_edge.ptFrom);LineObj.push_back(pose*_edge.ptTo);
    vector<Point3f> LineCam; LineCam.push_back(poseCam*LineObj[0]);LineCam.push_back(poseCam*LineObj[1]);
    
    vector<Point2f> vprojVertices;
    projectPoints(LineObj, poseCam.rvec(), poseCam.translation(), cameraMatrix, distCoeffs, vprojVertices);
    //if(LineCam[0].z>0 && LineCam[1].z>0)
        line(img, vprojVertices[0], vprojVertices[1], Scalar(0,0,255), 2);
    
    
    
}

float rotationVSfrontoparallel(const Vec3d &rvec)
{
    Affine3d cameraRotPose = Affine3d().rotate(rvec);
    Point3f frontoPerp(0.,0,1);
    
    //express frontoPerp in cam
    Point3f frontoPerpCam=cameraRotPose*frontoPerp;
    //get angle with frontoperp from arcsin of length of frontoPerpCam on x,y
    return asin(sqrt(frontoPerpCam.x*frontoPerpCam.x+frontoPerpCam.y*frontoPerpCam.y));
}

bool compareByDiscriminativePower(const DetectionGH &a, const DetectionGH &b)
{
    return a.discriminativePower > b.discriminativePower;
}

bool Object3D::getPose(const IntrinsicCalibration &_mCalib, vector<DetectionGH> mMatches, Affine3d &robotPose, bool init) const
{
    //if don t have 4 measures then problem not solvable
    if(mMatches.size() < 4)
        return false;
    
    //sort matches with respect to discriminative power to ease ransac later
    //put higher discriminative detections first
    std::sort(mMatches.begin(), mMatches.end(), compareByDiscriminativePower);
    
    //create list of 3d points corresponding to detected projections
    vector<Point3f> detectedVertices;
    for(unsigned int i=0;i<mMatches.size();i++)detectedVertices.push_back(mVertices[mMatches[i].id]);
    
    //use opencv function
    //get previous position if there is any
    Vec3d rvec_prec,tvec_prec;
    if(!init)
    {
        rvec_prec=robotPose.rvec();
        tvec_prec=robotPose.translation();
    }
    
    //do a kind of ransac: try different subset to compute pose util find that more than majority agrees
    //if not consider tracker lost
    int nbBasePnp=4;//take four points out of set
    unsigned int pointers[nbBasePnp];
    for(int i=0;i<nbBasePnp;i++)pointers[i]=i;//set first pointers as first elements of list
    
    while(1)
    {
        //want to test all the unsorted subsets of 4 points
        Vec3d rvec,tvec;
        if(!init)    {rvec=rvec_prec;tvec=tvec_prec;}
        
        //create vectors corresponding to subset
        vector<Point3f> subsetVertices;
        vector<Point2f> subsetProjections;
        for(int i=0;i<nbBasePnp;i++)
        {
            subsetVertices.push_back(detectedVertices[pointers[i]]);
            subsetProjections.push_back(mMatches[pointers[i]].position);
        }
        
        
        //compute pose with subset
        cv::solvePnP(subsetVertices, subsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, !init);
        //cv::solvePnPRansac(detectedVertices, pointPositions, cameraMatrix, distCoeffs, rvec, tvec);
        
        //check if estimated transformation is possible (if rotation with respect to image plan > max rotation GH training
        //then would not be able to do this association using GH)
        if(rotationVSfrontoparallel(rvec)<M_PI/4)
        {
            
            //check how many points agree
            float threshold_proj=5.;//set error max to 5 pixels
            unsigned int nbPointAgree=0;
            vector<Point2f> vProjPoints;//project all points
            projectPoints(detectedVertices, rvec, tvec, _mCalib.cameraMatrix, _mCalib.distCoeffs, vProjPoints);
            for(unsigned int i=0;i<mMatches.size();i++)
                if(norm(vProjPoints[i]-mMatches[i].position)<threshold_proj)
                    nbPointAgree++;
            
            //if more than majority agrees then fine, recompute objects pose with all points which agree and return it
            //might be better to instead do a tuckey optimisation with all points, as their might be points rejected with
            //first subset which after some refining would be back in...
            //can also run several iterations of the previous selection and following optimisation
            if(nbPointAgree>mMatches.size()/2)
            {
                vector<Point3f> newSubsetVertices;
                vector<Point2f> newSubsetProjections;
                for(unsigned int i=0;i<mMatches.size();i++)
                    if(norm(vProjPoints[i]-mMatches[i].position)<threshold_proj)
                    {
                        newSubsetVertices.push_back(detectedVertices[i]);
                        newSubsetProjections.push_back(mMatches[i].position);
                    }
                
                cv::solvePnP(newSubsetVertices, newSubsetProjections, _mCalib.cameraMatrix, _mCalib.distCoeffs, rvec, tvec, !init);
                robotPose=Affine3d(rvec,tvec);
                return true;
            }
        }
        
        //if not need to go to next subset:
        //try moving last pointer, if pointer reaches end, then have to move previous pointer
        //recursive if any pointer reaches its end position then moves it previous pointer and put current pointer after it
        //do until first point (pointers[0]) reaches the end (for pointers[0] corresponds to pointPositions.size-nbBasePnp)
        
        //to do that just count how many pointers are stuck at the end:
        int nbPtStuckAtEnd=0;
        for(int i=nbBasePnp-1;i>=0;i--)
            if(pointers[i]==mMatches.size()-nbBasePnp+i)
                nbPtStuckAtEnd++;
        
        if(nbPtStuckAtEnd==0)//if none stuck then can simply move last one
            pointers[nbBasePnp-1]++;
        else if(nbPtStuckAtEnd==nbBasePnp)//all of them are stuck and we didn t find anything good=> lost
            return false;
        else//some of them are at the end
        {
            //need to move the (nbBasePnp-1-nbPtStuckAtEnd)th pointer right and set the others to follow
            pointers[nbBasePnp-1-nbPtStuckAtEnd]++;
            for(int i=nbBasePnp-1-nbPtStuckAtEnd+1;i<nbBasePnp;i++)
                pointers[i]=pointers[i-1]+1;
        }
        
    }
}


Camera3dModel::Camera3dModel()
{
    vector<Point3f> mVerticesTemp;
    float scale=0.04;//make it about 5cm big
    float w=scale*0.4,h=scale*0.3,z=scale*0.8;
    
    //create vertices from dimensions
    mVerticesTemp.push_back(Point3f(0,0,0));
    mVerticesTemp.push_back(Point3f(w,h,z));
    mVerticesTemp.push_back(Point3f(w,-h,z));
    mVerticesTemp.push_back(Point3f(-w,-h,z));
    mVerticesTemp.push_back(Point3f(-w,h,z));
    
    //from center to image plane
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[1]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[2]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[3]));
    mEdges.push_back(ModelEdge(mVerticesTemp[0],mVerticesTemp[4]));
    
    //image plane
    mEdges.push_back(ModelEdge(mVerticesTemp[1],mVerticesTemp[2]));
    mEdges.push_back(ModelEdge(mVerticesTemp[2],mVerticesTemp[3]));
    mEdges.push_back(ModelEdge(mVerticesTemp[3],mVerticesTemp[4]));
    mEdges.push_back(ModelEdge(mVerticesTemp[4],mVerticesTemp[1]));
}

    
bool Object3D::getPoseFromBlobs(const std::vector<cv::KeyPoint> &blobs,const IntrinsicCalibration& _mCalib, cv::Affine3d& robotPose, bool init)
{
    Grouping mGrouping;
    //first extract blobs groups
    vector<BlobPair> blobPairs;
    mGrouping.getPairsFromBlobs(blobs,blobPairs);
    
    vector<BlobTriplet> blobTriplets;
    mGrouping.getTripletsFromPairs(blobs,blobPairs,blobTriplets);
    setClockwiseDirectionToTriplets(blobs,blobTriplets);
    
    vector<BlobQuadruplets> blobQuadriplets;
    mGrouping.getQuadripletsFromTriplets(blobTriplets,blobQuadriplets,true);//here we remove the triplets in quads
    setClockwiseDirectionToQuadruplets(blobs,blobQuadriplets);
    
    //then compute all the poses from all the possible P3P
    vector<cv::Affine3d> poseHypothesis;
    //convert all blob positions to meter coordinates
    vector<cv::Point2f> coordImageSpaceBlobs;
    for(int i=0;i<blobs.size();i++)
        coordImageSpaceBlobs.push_back(toMeters(_mCalib.cameraMatrix,blobs[i].pt));

    //for all the triplets compute pose hypothesis from Groups of 3 points from model
    //std::cout<<"compute pose hypotheses"<<std::endl;
    for(int i=0;i<mGroup3s.size();i++)
        for(int j=0;j<blobTriplets.size();j++)
    {
        //std::cout<<"\t triplet "<<j<<std::endl;
        //std::cout<<"\t group "<<i<<std::endl;

        BlobTriplet &b_triplet = blobTriplets[j];
        ModelTriplet &m_triplet = mGroup3s[i];

        vector<cv::Point3f> Model3DPoints;
        for(int p=0;p<3;p++)
            Model3DPoints.push_back(mVertices[m_triplet.ids[p]]);

        //as both tripplets are directed we have 3 possible matches between them
        for(int k=0;k<3;k++)
        {
            vector<cv::Point2f> projected2DPoints;
            for(int p=0;p<3;p++)
                projected2DPoints.push_back(coordImageSpaceBlobs[b_triplet.ids[ (p+k)%3 ]]);

            //compute the 4 hypothesis (pose cam to world=>have to inverse them)
            vector<cv::Affine3d> poseHypothesisP3Pinv = computeP3P(projected2DPoints,Model3DPoints);

            //put the hypotheses in accumulation
            for(int h=0;h<poseHypothesisP3Pinv.size();h++)
            {
                //get inverse:
                cv::Affine3d poseHypothesisP3P=poseHypothesisP3Pinv[h].inv();

                //reject hypothesis where object is behind camera
                Point3f Origin(0,0,0);
                Point3f CenterObj = poseHypothesisP3P*Origin;

                if(CenterObj.z>0)
                    poseHypothesis.push_back(poseHypothesisP3P);
            }
        }
    }

    //for all the quadruplets compute pose hypothesis from Groups of 4 points from model
    //std::cout<<"compute pose hypotheses"<<std::endl;
    for(int i=0;i<mGroup4s.size();i++)
        for(int j=0;j<blobQuadriplets.size();j++)
    {
        //std::cout<<"\t triplet "<<j<<std::endl;
        //std::cout<<"\t group "<<i<<std::endl;

        BlobQuadruplets &b_quad = blobQuadriplets[j];
        ModelQuadruplet &m_quad = mGroup4s[i];

        vector<cv::Point3f> Model3DPoints;
        for(int p=0;p<3;p++)//only use 3 points for pose computation (as the 4 are coplanar)
            Model3DPoints.push_back(mVertices[m_quad.ids[p]]);

        //as both quadruplets are directed we have 4 possible matches between them
        for(int k=0;k<4;k++)
        {
            vector<cv::Point2f> projected2DPoints;
            for(int p=0;p<3;p++)
                projected2DPoints.push_back(coordImageSpaceBlobs[b_quad.ids[ (p+k)%4 ]]);

            //compute the 4 hypothesis (pose cam to world=>have to inverse them)
            vector<cv::Affine3d> poseHypothesisP3Pinv = computeP3P(projected2DPoints,Model3DPoints);

            //put the hypotheses in accumulation
            for(int h=0;h<poseHypothesisP3Pinv.size();h++)
            {
                //get inverse:
                cv::Affine3d poseHypothesisP3P=poseHypothesisP3Pinv[h].inv();

                //reject hypothesis where object is behind camera
                Point3f Origin(0,0,0);
                Point3f CenterObj = poseHypothesisP3P*Origin;

                if(CenterObj.z>0)
                    poseHypothesis.push_back(poseHypothesisP3P);
            }
        }
    }


    //std::cout<<"end compute pose hypotheses"<<std::endl;
    
    //sort out the hypothesis which are impossible
    //use IMU
    //use temporal consistency

    //project model using remaining hypothesis onto image
    Mat projectionAccu(_mCalib.imageSize.height,_mCalib.imageSize.width,CV_32FC1,Scalar(0));
    //each model blob will be proejcted on the image and it will vote for a blob to be there
    //(then can read, for each pose, how many votes agreed)
    //first need to define a noise on the measure
    float sigma_meas = 5.;
    //for each pose do the update
    std::cout<<poseHypothesis.size()<<" hypotheses"<<std::endl;
    for(int h=0;h<poseHypothesis.size();h++)
    {
        //projected vertices
        vector<Point2f> vprojVertices;
        //projectPoints(mVertices, poseHypothesis[h].rvec(), poseHypothesis[h].translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVertices);

        for(int i=0;i<mGroup3s.size();i++)
        {
            std::vector<cv::Point3f> mVerticesInGroup;
            for(int j=0;j<3;j++)mVerticesInGroup.push_back(mVertices[mGroup3s[i].ids[j]]);

            //project them
            vector<Point2f> vprojVerticesInGroup;
            projectPoints(mVerticesInGroup, poseHypothesis[h].rvec(), poseHypothesis[h].translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVerticesInGroup);

            //add them if direction is good
            if(testDirectionGroup(vprojVerticesInGroup[0],vprojVerticesInGroup[1],vprojVerticesInGroup[2]))
                for(int j=0;j<3;j++)vprojVertices.push_back(vprojVerticesInGroup[j]);
        }
        //same for groups of 4
        for(int i=0;i<mGroup4s.size();i++)
        {
            std::vector<cv::Point3f> mVerticesInGroup;
            for(int j=0;j<4;j++)mVerticesInGroup.push_back(mVertices[mGroup4s[i].ids[j]]);

            //project them
            vector<Point2f> vprojVerticesInGroup;
            projectPoints(mVerticesInGroup, poseHypothesis[h].rvec(), poseHypothesis[h].translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVerticesInGroup);

            //add them if direction is good
            if(testDirectionGroup(vprojVerticesInGroup[0],vprojVerticesInGroup[1],vprojVerticesInGroup[2]))
                for(int j=0;j<4;j++)vprojVertices.push_back(vprojVerticesInGroup[j]);
        }

        //update projectionAccu
        int support_normal = int(3.*sigma_meas);
        for(int i=0;i<vprojVertices.size();i++)
        {
            //std::cout<<"vprojVertices ["<<i<<"] = "<< vprojVertices[i] <<std::endl;
            if(vprojVertices[i].x>-support_normal && vprojVertices[i].y>-support_normal && 
                vprojVertices[i].x<_mCalib.imageSize.width+support_normal && vprojVertices[i].y<_mCalib.imageSize.height+support_normal)
                for(int x=vprojVertices[i].x-support_normal; x <vprojVertices[i].x+support_normal; x++)
                    for(int y=vprojVertices[i].y-support_normal; y <vprojVertices[i].y+support_normal; y++)
                       if(x>=0 && y>=0 && x<_mCalib.imageSize.width && y<_mCalib.imageSize.height)
                        {
                            float dist_squared = ((float)x-vprojVertices[i].x)*((float)x-vprojVertices[i].x)
                                            +((float)y-vprojVertices[i].y)*((float)y-vprojVertices[i].y);
                            float normal = exp(-dist_squared/(sigma_meas*sigma_meas))/(sigma_meas*sqrt(2*M_PI));

                            projectionAccu.at<float>(y,x)+=normal;
                        }
        }
    }

    //display projectionAccu for testing
    /*double min;
    double max;
    cv::minMaxIdx(projectionAccu, &min, &max);
    cv::Mat adjMap;
    cv::convertScaleAbs(projectionAccu, adjMap, 255 / max);
    cv::imshow("Out", adjMap);
    cv::waitKey();*/

    //now read nbVotes for each pose
    float nbVotesMax=-1;
    float idBestHypothesis=0;

    //keep nbHypoPerTime best hypothesis
    //PoseHypothesisSet *bestHypothesis=new PoseHypothesisSet[nbHypoPerTime];
    //int idMinScoreSet = 0; //which of the hypos to replace next

    for(int h=0;h<poseHypothesis.size();h++)
    {
        //projected vertices
        vector<Point2f> vprojVertices;
        projectPoints(mVertices, poseHypothesis[h].rvec(), poseHypothesis[h].translation(), _mCalib.cameraMatrix, _mCalib.distCoeffs, vprojVertices);

        float nbVotes = 0;
        for(int i=0;i<vprojVertices.size();i++)
        {
            int x_read = round(vprojVertices[i].x);
            int y_read = round(vprojVertices[i].y);
            if(x_read>=0 && y_read>=0 && x_read<_mCalib.imageSize.width && y_read<_mCalib.imageSize.height)
                nbVotes+=projectionAccu.at<float>(y_read,x_read);
        }

        if(nbVotes>nbVotesMax)
        {
            nbVotesMax=nbVotes;
            idBestHypothesis=h;
        }

        /*if(nbVotes > bestHypothesis[idMinScoreSet].score)
        {
            //change hyp
            bestHypothesis[idMinScoreSet].score = nbVotes;
            bestHypothesis[idMinScoreSet].pose = poseHypothesis[h];

            //upadte idMinScoreSet
            idMinScoreSet = 0;
            float scoreMinBuff = bestHypothesis[idMinScoreSet].score;

            for(int i=1;i<nbHypoPerTime;i++)
                if(bestHypothesis[i].score<scoreMinBuff)
                {
                    idMinScoreSet = i;
                    scoreMinBuff = bestHypothesis[idMinScoreSet].score;
                }
        }*/
    }

    //temporal consistency:
    //we keep track of the N best hypothesis for the M previous times
    //sometimes the object might have been missed=> also keep a boolean
    //for each time weither the object was estimated found or not
    //if not found then should not use the time for discrimiation

    //=>start with history with only unfound object
    //find best hypothesis, then could say object found while it is not...
    //=> probably should not use boolean as we don't really know it at any time

    //start with random pose hypothesis in history
    //=> can just check how many hypothesis agree with current one
    //and motion model (eg object has to move of 10 cm at each time step)
    //for that can cumulate votes as : for each hypo, go through t-1 hypos and 
    //add score of corresponding hypos weighted by distance with original hypo.

    
    //now if all the blobs were seen then we d have at least score of nbVertices*gaussian_meas(0)
    //we can set a threshold on that to define if we found the object or not
    bool found = false;
    std::cout<<"nbVotesMax = "<<nbVotesMax<<std::endl;
    if(nbVotesMax > 0.5*mVertices.size())
    {
        found = true;
        robotPose=poseHypothesis[idBestHypothesis];

        Point3f Origin(0,0,0);
        Point3f CenterObj = robotPose*Origin;
        std::cout<<"z = "<<CenterObj.z<<std::endl;
    }

    //define coarse motion model
    /*float sigma_translation_per_time = 0.1;
    //for each current hypo check how much history agrees with it
    int idBestCurrentHypo = 0;
    float bestScoreWithTime = 0;
    for(int h=0;h<nbHypoPerTime && h<poseHypothesis.size();h++)
    {
        //score of agreement with current hypo and history
        float score = bestHypothesis[h].score;

        for(int t=0;t<lengthHistory;t++)
            for(int h2=0;h2<nbHypoPerTime;h2++)
            {
                //add score of history hypo weighted by distance with current
                Vec3d dist_translation = bestHypothesis[h].pose.translation() - PoseHypothesisHistory[t][h2].pose.translation();
                float dist_norm = norm(dist_translation);
                float sigma_c = sigma_translation_per_time*(t+1);
                float weight = exp(-dist_norm*dist_norm/(2.*sigma_c*sigma_c))/(sigma_c*sqrt(2)*M_PI);
                score+= weight*PoseHypothesisHistory[t][h2].score;
            }

        std::cout<<"\tscore ["<<h<<"] = "<<score<<std::endl;
        if(score>bestScoreWithTime)
        {
            bestScoreWithTime=score;
            idBestCurrentHypo=h;
        }
    }

    //rolling buffer on hypos
    for(int t=lengthHistory-1;t>0;t--)
        for(int h=0;h<nbHypoPerTime;h++)
            PoseHypothesisHistory[t][h] = PoseHypothesisHistory[t-1][h];
    for(int h=0;h<nbHypoPerTime;h++)
        PoseHypothesisHistory[0][h] = bestHypothesis[h];
    
    bool found = true;
    robotPose=bestHypothesis[idBestCurrentHypo].pose;

    std::cout<<"score = "<< bestScoreWithTime<<std::endl;
    
    delete[] bestHypothesis;*/
    return found;
}
    


ThymioBlobModel::ThymioBlobModel()
{
    //vertices in cm, origin is on corner blob
    /*mVertices.push_back(Point3f(0,5.6,3.2));
    mVertices.push_back(Point3f(0,4.8,3.2));
    mVertices.push_back(Point3f(0,0.8,3.2));
    mVertices.push_back(Point3f(0,0,3.2));
    
    mVertices.push_back(Point3f(0.8,4.8,3.2));
    mVertices.push_back(Point3f(0.8,0.8,3.2));
    mVertices.push_back(Point3f(0.8,0,3.2));
    
    mVertices.push_back(Point3f(8.,4.8,3.2));
    mVertices.push_back(Point3f(8.,0.8,3.2));
    mVertices.push_back(Point3f(8.,0,3.2));
    
    mVertices.push_back(Point3f(8.8,5.6,3.2));
    mVertices.push_back(Point3f(8.8,4.8,3.2));
    mVertices.push_back(Point3f(8.8,0.8,3.2));
    mVertices.push_back(Point3f(8.8,0,3.2));*/
    
    //top of the robot
    //model definition:
    // 10             0
    // 11  7       4  1
    //
    // 12  8       5  2
    // 13  9       6  3
    
    //set everything in centimeters and will divide later to convert in meters
    mVertices.push_back(Point3f(8.8,5.6,3.2));
    mVertices.push_back(Point3f(8.8,4.8,3.2));
    mVertices.push_back(Point3f(8.8,0.8,3.2));
    mVertices.push_back(Point3f(8.8,0,3.2));
    
    mVertices.push_back(Point3f(8,4.8,3.2));
    mVertices.push_back(Point3f(8,0.8,3.2));
    mVertices.push_back(Point3f(8,0,3.2));
    
    mVertices.push_back(Point3f(0.8,4.8,3.2));
    mVertices.push_back(Point3f(0.8,0.8,3.2));
    mVertices.push_back(Point3f(0.8,0,3.2));
    
    mVertices.push_back(Point3f(0.,5.6,3.2));
    mVertices.push_back(Point3f(0.,4.8,3.2));
    mVertices.push_back(Point3f(0.,0.8,3.2));
    mVertices.push_back(Point3f(0.,0,3.2));
    
    //define groups in clockwise order
    mGroup3s.push_back(ModelTriplet(0,1,4));
    mGroup3s.push_back(ModelTriplet(10,7,11));
    
    mGroup4s.push_back(ModelQuadruplet(5,2,3,6));
    mGroup4s.push_back(ModelQuadruplet(12,8,9,13));

    //set them in meter and center them
    for(int v=0;v<14;v++)
        mVertices[v]=(mVertices[v]-Point3f(4.4,2.,0))/10.;
        //mVertices[v]=(mVertices[v]-Point3f(4.4,2.8,0))/10.;

    //back of robot again put in centimeters and origin as left down vertice
    //     16           19
    //  15                  18
    //     14           17
    mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(-0.7,0.,0.675));
    mVertices.push_back(Point3f(0.,0.,1.35));
    
    mVertices.push_back(Point3f(8.4,0.,0.));
    mVertices.push_back(Point3f(9.1,0.,0.675));
    mVertices.push_back(Point3f(8.4,0.,1.35));

    for(int v=14;v<20;v++)
        mVertices[v]=(mVertices[v]-Point3f(4.1,2.9,-0.3))/10.;

    mGroup3s.push_back(ModelTriplet(14,15,16));
    mGroup3s.push_back(ModelTriplet(19,18,17));

    //side left
    // 21
    // 
    // 20   22
    mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(0.,0.,0.95));
    mVertices.push_back(Point3f(0.,-0.95,0));
    for(int v=20;v<23;v++)
        mVertices[v]=(mVertices[v]-Point3f(5.5,-5.1,0.45))/10.;
    mGroup3s.push_back(ModelTriplet(20,21,22));
    
    //right left
    //      25
    // 
    // 24   23
    mVertices.push_back(Point3f(0.,0.,0.));
    mVertices.push_back(Point3f(0.,-0.95,0));
    mVertices.push_back(Point3f(0.,0.,0.95));
    for(int v=23;v<26;v++)
        mVertices[v]=(mVertices[v]-Point3f(-5.5,-5.1,0.45))/10.;
    mGroup3s.push_back(ModelTriplet(23,24,25));


  



    
    
    //if want to display edges
    vector<Point3f> mVerticesTop;
    //create vertices from dimensions
    mVerticesTop.push_back(Point3f(-0.65,7.75,3.2));
    mVerticesTop.push_back(Point3f(-0.65,-0.65,3.2));
    mVerticesTop.push_back(Point3f(9.5,-0.65,3.2));
    mVerticesTop.push_back(Point3f(9.5,7.75,3.2));
    mVerticesTop.push_back(Point3f(7.,9.4,3.2));
    mVerticesTop.push_back(Point3f(5.,9.7,3.2));
    mVerticesTop.push_back(Point3f(2.5,9.4,3.2));
    
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesTop.size();v++)
        mVerticesTop[v]=(mVerticesTop[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesTop.size();v++)
        mEdges.push_back(ModelEdge(mVerticesTop[v],mVerticesTop[(v+1)%mVerticesTop.size()]));
    
    //if want to display edges of bottom part
    vector<Point3f> mVerticesBottom;
    //create vertices from dimensions
    mVerticesBottom.push_back(Point3f(-0.65,7.75,-1.2));
    mVerticesBottom.push_back(Point3f(-0.65,-0.65,-1.2));
    mVerticesBottom.push_back(Point3f(9.5,-0.65,-1.2));
    mVerticesBottom.push_back(Point3f(9.5,7.75,-1.2));
    mVerticesBottom.push_back(Point3f(7.,9.4,-1.2));
    mVerticesBottom.push_back(Point3f(5.,9.7,-1.2));
    mVerticesBottom.push_back(Point3f(2.5,9.4,-1.2));
    
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesBottom.size();v++)
        mVerticesBottom[v]=(mVerticesBottom[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesBottom.size();v++)
        mEdges.push_back(ModelEdge(mVerticesBottom[v],mVerticesBottom[(v+1)%mVerticesBottom.size()]));
    
    //vertical edges
    vector<Point3f> mVerticesVertical;
    //create vertices from dimensions
    mVerticesVertical.push_back(Point3f(-0.65,7.75,3.2));
    mVerticesVertical.push_back(Point3f(-0.65,7.75,-1.2));
    mVerticesVertical.push_back(Point3f(-0.65,-0.65,3.2));
    mVerticesVertical.push_back(Point3f(-0.65,-0.65,-1.2));
    mVerticesVertical.push_back(Point3f(9.5,-0.65,3.2));
    mVerticesVertical.push_back(Point3f(9.5,-0.65,-1.2));
    mVerticesVertical.push_back(Point3f(9.5,7.75,3.2));
    mVerticesVertical.push_back(Point3f(9.5,7.75,-1.2));
    //set them in meter and center them
    for(unsigned int v=0;v<mVerticesVertical.size();v++)
        mVerticesVertical[v]=(mVerticesVertical[v]-Point3f(4.4,2.8,0))/100.;
    
    //from center to image plane
    for(unsigned int v=0;v<mVerticesVertical.size()/2;v++)
        mEdges.push_back(ModelEdge(mVerticesVertical[2*v],mVerticesVertical[2*v+1]));
    
}

}
