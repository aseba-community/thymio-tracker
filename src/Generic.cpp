
#include "Generic.hpp"

using namespace cv;
using namespace std;

namespace thymio_tracker
{

void readCalibrationFromFileStorage(cv::FileStorage &fs, IntrinsicCalibration &calibration)
{
    fs["camera_matrix"] >> calibration.cameraMatrix;
    fs["distortion_coefficients"] >> calibration.distCoeffs;
    calibration.imageSize.width = (int) fs["image_width"];
    calibration.imageSize.height = (int) fs["image_height"];
    
}


void resizeCameraMatrix(Mat& _cameraMatrix, const Size& sFrom, const Size& sTo)
{
    float ratiox=(float)sTo.width/sFrom.width;
    float ratioy=(float)sTo.height/sFrom.height;
    for(int j=0; j<3; j++)
    {
        _cameraMatrix.at<double>(0,j)=_cameraMatrix.at<double>(0,j)*ratiox;
        _cameraMatrix.at<double>(1,j)=_cameraMatrix.at<double>(1,j)*ratioy;
    }
}

void rescaleCalibration(IntrinsicCalibration &calibration, const cv::Size& sTo)
{
    resizeCameraMatrix(calibration.cameraMatrix, calibration.imageSize, sTo);
    calibration.imageSize=sTo;
}

Point2f toPixels(const Mat& _cameraMatrix, const Point2f& _m)
{
    Point2f px;
    px.x=_cameraMatrix.at<double>(0,0)*_m.x+_cameraMatrix.at<double>(0,2);
    px.y=_cameraMatrix.at<double>(1,1)*_m.y+_cameraMatrix.at<double>(1,2);
    return px;
}

Point2f toMeters(const Mat& _cameraMatrix, const Point2f& _x)
{
    Point2f pm;
    pm.x=(_x.x-_cameraMatrix.at<double>(0,2))/_cameraMatrix.at<double>(0,0);
    pm.y=(_x.y-_cameraMatrix.at<double>(1,2))/_cameraMatrix.at<double>(1,1);
    return pm;
}

Point2f toMeters(const Mat& _cameraMatrix, const Point2i& _x)
{
    Point2f pm;
    pm.x=((float)_x.x-_cameraMatrix.at<double>(0,2))/_cameraMatrix.at<double>(0,0);
    pm.y=((float)_x.y-_cameraMatrix.at<double>(1,2))/_cameraMatrix.at<double>(1,1);
    return pm;
}

Point2f toImagePlane(const Point3f& _x)
{
    return Point2f(_x.x/_x.z,_x.y/_x.z);
}

Mat ProjectZ1_Jac_Dp(const Point3f& mvLastDistCam)
{
    Mat res(2, 6, DataType<float>::type);
    float invZ=1./mvLastDistCam.z;
    float x=mvLastDistCam.x*invZ;
    float y=mvLastDistCam.y*invZ;
    
    res.at<float>(0,0)= invZ;   res.at<float>(0,1)= 0.;         res.at<float>(0,2)= -x*invZ;
    res.at<float>(0,3)= -x*y;   res.at<float>(0,4)= (1+x*x);    res.at<float>(0,5)= -y;
    res.at<float>(1,0)= 0.;     res.at<float>(1,1)= invZ;       res.at<float>(1,2)= -y*invZ;
    res.at<float>(1,3)= -(1+y*y);res.at<float>(1,4)= x*y;       res.at<float>(1,5)= x;
    return res;
}

cv::Point2f Pointxy(const cv::Point3f& _m){return cv::Point2f(_m.x,_m.y);}


bool testDirectionBasis(Point2f basis1,Point2f basis2)
{
    float crossProd=basis1.x*basis2.y-basis1.y*basis2.x;
    return crossProd>0;
}
bool testDirectionGroup(cv::Point2f v1,cv::Point2f v2,cv::Point2f v3)
{
    return testDirectionBasis(v2-v1,v3-v1);
}

}
