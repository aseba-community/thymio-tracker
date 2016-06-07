
#include "TrackingFcts.hpp"

using namespace cv;
using namespace std;

namespace thymio_tracker
{

void printMat(const CvMat* mat)
{
    //IplImage img=mat;
    cv::Mat matt=cvarrToMat(mat);
    std::cout<<mat->rows<<std::endl;
    std::cout<<mat->cols<<std::endl;
    std::cout<<matt.channels()<<std::endl;
    for(int i=0; i<mat->rows; i++)
    {
        for(int j=0; j<mat->cols; j++)
        {   
            for(int k=0; k<matt.channels(); k++)
                std::cout<<cvGet2D(mat,i,j).val[k]<<"\t";
            std::cout<<std::endl;
        }
        std::cout<<std::endl<<std::endl;
    }

}

//multiply matrix by a matrix with one element only
void mulErrMat(const CvMat* mat,const CvMat* s,CvMat* matr)
{
    //score has shape (1,nbMatch,1)
    //err has shape (1,nbMatch,2)
    cv::Mat mat_r=cvarrToMat(mat);
    cv::Mat mat_w=cvarrToMat(matr);
    int nbChannel = mat_r.channels();

    for(int i=0; i<mat->rows; i++)
    {
        //float *ptr_r = mat_r.at<float>
        for(int j=0; j<mat->cols; j++)
        {   
            for(int k=0; k<mat_r.channels(); k++)
                mat_w.ptr<double>(i)[j*nbChannel+k] = cvGet2D(s,i,j).val[0]*mat_r.ptr<double>(i)[j*nbChannel+k];
        }
    }

}

void mulJacMat(const CvMat* mat,const CvMat* s,CvMat* matr)
{
    //score has shape (1,nbMatch,1)
    //Jac has shape (2*nbMatch,6,1)
    cv::Mat mat_r=cvarrToMat(mat);
    cv::Mat mat_w=cvarrToMat(matr);

    for(int i=0; i<s->cols; i++)
        for(int i2=0; i2<2; i2++)
    {
        for(int j=0; j<mat->cols; j++)
        {   
            mat_w.ptr<double>(2*i+i2)[j] = cvGet2D(s,0,i).val[0]*mat_r.ptr<double>(2*i+i2)[j];
        }
    }

}


void cvFindPoseScaled( const CvMat* objectPoints,
                  const CvMat* imagePoints, const CvMat* scores, const CvMat* A,
                  const CvMat* distCoeffs, CvMat* rvec, CvMat* tvec)
{
    const int max_iter = 20;
    Ptr<CvMat> matM, _m, _mn, matL, _s;

    int count;
    double a[9], ar[9]={1,0,0,0,1,0,0,0,1};
    CvScalar Mc;
    double param[6];
    CvMat matA = cvMat( 3, 3, CV_64F, a );
    CvMat _Ar = cvMat( 3, 3, CV_64F, ar );
    CvMat _r = cvMat( 3, 1, CV_64F, param );
    CvMat _t = cvMat( 3, 1, CV_64F, param + 3 );
    CvMat _param = cvMat( 6, 1, CV_64F, param );
    CvMat _dpdr, _dpdt;

    CV_Assert( CV_IS_MAT(objectPoints) && CV_IS_MAT(imagePoints) &&
        CV_IS_MAT(A) && CV_IS_MAT(rvec) && CV_IS_MAT(tvec) );

    count = MAX(objectPoints->cols, objectPoints->rows);
    matM.reset(cvCreateMat( 1, count, CV_64FC3 ));
    _m.reset(cvCreateMat( 1, count, CV_64FC2 ));

    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );
    cvConvert( A, &matA );


    CV_Assert( (CV_MAT_DEPTH(rvec->type) == CV_64F || CV_MAT_DEPTH(rvec->type) == CV_32F) &&
        (rvec->rows == 1 || rvec->cols == 1) && rvec->rows*rvec->cols*CV_MAT_CN(rvec->type) == 3 );

    CV_Assert( (CV_MAT_DEPTH(tvec->type) == CV_64F || CV_MAT_DEPTH(tvec->type) == CV_32F) &&
        (tvec->rows == 1 || tvec->cols == 1) && tvec->rows*tvec->cols*CV_MAT_CN(tvec->type) == 3 );

    _mn.reset(cvCreateMat( 1, count, CV_64FC2 ));

    // normalize image points
    // (unapply the intrinsic matrix transformation and distortion)
    cvUndistortPoints( _m, _mn, &matA, distCoeffs, 0, &_Ar );

    CvMat _r_temp = cvMat(rvec->rows, rvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
    CvMat _t_temp = cvMat(tvec->rows, tvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3);
    cvConvert( rvec, &_r_temp );
    cvConvert( tvec, &_t_temp );
    
    

    cvReshape( matM, matM, 3, 1 );//3 = number of channels, 1=number of rows
    cvReshape( _mn, _mn, 2, 1 );

    // refine extrinsic parameters using iterative algorithm
    CvLevMarq solver( 6, count*2, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,max_iter,FLT_EPSILON), true);
    cvCopy( &_param, solver.param );

    for(;;)
    {
        CvMat *matJ = 0, *_err = 0;
        const CvMat *__param = 0;
        bool proceed = solver.update( __param, matJ, _err );
        cvCopy( __param, &_param );
        if( !proceed || !_err )
            break;
        cvReshape( _err, _err, 2, 1 );

        if( matJ )
        {
            cvGetCols( matJ, &_dpdr, 0, 3 );
            cvGetCols( matJ, &_dpdt, 3, 6 );
            cvProjectPoints2( matM, &_r, &_t, &matA, distCoeffs,
                              _err, &_dpdr, &_dpdt, 0, 0, 0 );
            //in _err just get the projection of the points now and 
            mulJacMat(matJ,scores,matJ);
        }
        else
        {
            cvProjectPoints2( matM, &_r, &_t, &matA, distCoeffs,
                              _err, 0, 0, 0, 0, 0 );
        }
        cvSub(_err, _m, _err);//now get the error really
        mulErrMat(_err, scores, _err);

        cvReshape( _err, _err, 1, 2*count );
    }
    cvCopy( solver.param, &_param );

    _r = cvMat( rvec->rows, rvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
    _t = cvMat( tvec->rows, tvec->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3 );

    cvConvert( &_r, rvec );
    cvConvert( &_t, tvec );
}

bool robustPnp(InputArray opoints,InputArray ipoints,
    InputArray score, InputArray _cameraMatrix, InputArray _distCoeffs,
               OutputArray _rvec, OutputArray _tvec)
{
    //undistort points 
    CvMat c_objectPoints = opoints.getMat(), c_imagePoints = ipoints.getMat();
    CvMat c_score = score.getMat();
    CvMat c_cameraMatrix = _cameraMatrix.getMat(), c_distCoeffs = _distCoeffs.getMat();
    CvMat c_rvec = _rvec.getMat(), c_tvec = _tvec.getMat();
    
    cvFindPoseScaled(&c_objectPoints, &c_imagePoints, &c_score, &c_cameraMatrix,&c_distCoeffs,
                                     &c_rvec, &c_tvec);
    return true;
    
}

float MI(cv::Mat img, cv::Mat &templ, cv::Mat &mask)
{
    int nbBin = 8;
    //float Pm[nbBin],Pn[nbBin],Pmn[nbBin*nbBin];
    float *Pm=new float[nbBin];
    float *Pn=new float[nbBin];
    float *Pmn=new float[nbBin*nbBin];

    //init bins
    for(int i=0;i<nbBin;i++)
    {
        Pm[i]=0;
        Pn[i]=0;
        for(int j=0;j<nbBin;j++)
            Pmn[nbBin*i+j]=0;
    }

    //fill bins
    //NN
    /*int nbPix = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        uchar* pixel_mask = mask.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_m = img.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_n = templ.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img.cols; ++j)
        {

            uchar mask = *pixel_mask++;
            if (mask>0)
            {
                uchar m = *pixel_m++;
                uchar n = *pixel_n++;

                uchar bm = (int)(nbBin*(float)m/256.);
                uchar bn = (int)(nbBin*(float)n/256.);

                Pm[bm]++;
                Pn[bn]++;
                Pmn[nbBin*bm+bn]++;
                nbPix++;
            }
        }
    }*/

    int nbPix = 0;
    for (int i = 0; i < img.rows; ++i)
    {
        uchar* pixel_mask = mask.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_m = img.ptr<uchar>(i);  // point to first color in row
        uchar* pixel_n = templ.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img.cols; ++j)
        {

            uchar mask = *pixel_mask++;
            if (mask>0)
            {
                uchar m = *pixel_m++;
                uchar n = *pixel_n++;

                float bm = (float)(nbBin-1.)*(float)m/256.;
                float bn = (float)(nbBin-1.)*(float)n/256.;

                int Ebm = (int)bm;
                int Ebn = (int)bn;
                float em = bm-(float)Ebm;
                float en = bn-(float)Ebn;

                Pm[Ebm]  += 1.- em;
                Pm[Ebm+1]+= em;

                Pn[Ebn]  += 1.- en;
                Pn[Ebn+1]+= en;

                Pmn[nbBin*(Ebm)+Ebn] += (1.- em)*(1.- en);
                Pmn[nbBin*(Ebm+1)+Ebn] += (em)*(1.- en);
                Pmn[nbBin*(Ebm)+Ebn+1] += (1.- em)*(en);
                Pmn[nbBin*(Ebm+1)+Ebn+1] += (em)*(en);

                nbPix++;
            }
        }
    }
    //normalise
    //int nbPix = img.rows*img.cols;
    for(int i=0;i<nbBin;i++)
    {
        Pm[i]=Pm[i]/nbPix;
        Pn[i]=Pn[i]/nbPix;
        for(int j=0;j<nbBin;j++)
            Pmn[nbBin*i+j]=Pmn[nbBin*i+j]/nbPix;
    }

    //compute MI
    float res = 0;
   for(int i=0;i<nbBin;i++)
    {
        for(int j=0;j<nbBin;j++)
            if(Pmn[nbBin*i+j]>0 && Pm[i]>0 && Pn[j]>0)
                res += Pmn[nbBin*i+j]*log(Pmn[nbBin*i+j]/(Pm[i]*Pn[j]));
    }

    delete[] Pm;
    delete[] Pn;
    delete[] Pmn;

    return res;
}

void matchTemplateMI( cv::Mat img, cv::Mat &templ, cv::Mat &res, cv::Mat &mask)
{
    for(int i=0;i<res.size().height;i++)
        for(int j=0;j<res.size().width;j++)
        {
            Rect box = cv::Rect(j,i,templ.size().width,templ.size().height);
            res.at<float>(i,j) = MI(img(box),templ,mask);
        }
}

void parabolicRefinement(cv::Mat &curv,cv::Point maxLoc,cv::Point2f &maxLocF)
{
    if(maxLoc.x>0 && maxLoc.y>0 && maxLoc.x<curv.size().width-1 && maxLoc.y<curv.size().height-1)
        maxLocF = maxLoc;
    else
    {
        float centerVal = curv.at<float>(maxLoc.y,maxLoc.x);

        //fit parabola horizontally and vertically
        //for this simply get derivative x left and right and do linear regression
        float dx_l = centerVal - curv.at<float>(maxLoc.y,maxLoc.x-1);
        float dx_r = curv.at<float>(maxLoc.y,maxLoc.x+1) - centerVal;

        float dy_t = centerVal - curv.at<float>(maxLoc.y-1,maxLoc.x);
        float dy_b = curv.at<float>(maxLoc.y+1,maxLoc.x) - centerVal;

        float ex = -0.5 + dx_l / (dx_l-dx_r);
        float ey = -0.5 + dy_t / (dy_t-dy_b);

        maxLocF = cv::Point2f(maxLoc)+cv::Point2f(ex,ey);
    }
}           

}
