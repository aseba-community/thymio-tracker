
#include "Generic.hpp"
#include <Eigen/Eigen>

#include "P3P.hpp"

using namespace cv;
using namespace std;

namespace thymio_tracker
{

using namespace std;
using namespace Eigen;


std::vector<double> o4_roots( const Eigen::MatrixXd & p )
{
    double A = p(0,0);
    double B = p(1,0);
    double C = p(2,0);
    double D = p(3,0);
    double E = p(4,0);
    
    double A_pw2 = A*A;
    double B_pw2 = B*B;
    double A_pw3 = A_pw2*A;
    double B_pw3 = B_pw2*B;
    double A_pw4 = A_pw3*A;
    double B_pw4 = B_pw3*B;
    
    double alpha = -3*B_pw2/(8*A_pw2)+C/A;
    double beta = B_pw3/(8*A_pw3)-B*C/(2*A_pw2)+D/A;
    double gamma = -3*B_pw4/(256*A_pw4)+B_pw2*C/(16*A_pw3)-B*D/(4*A_pw2)+E/A;
    
    double alpha_pw2 = alpha*alpha;
    double alpha_pw3 = alpha_pw2*alpha;
    
    std::complex<double> P (-alpha_pw2/12-gamma,0);
    std::complex<double> Q (-alpha_pw3/108+alpha*gamma/3-pow(beta,2)/8,0);
    std::complex<double> R = -Q/2.0+sqrt(pow(Q,2.0)/4.0+pow(P,3.0)/27.0);
    
    std::complex<double> U = pow(R,(1.0/3.0));
    std::complex<double> y;
    
    if (U.real() == 0)
        y = -5.0*alpha/6.0-pow(Q,(1.0/3.0));
    else
        y = -5.0*alpha/6.0-P/(3.0*U)+U;
    
    std::complex<double> w = sqrt(alpha+2.0*y);
    
    std::vector<double> realRoots;
    std::complex<double> temp;
    temp = -B/(4.0*A) + 0.5*(w+sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
    realRoots.push_back(temp.real());
    temp = -B/(4.0*A) + 0.5*(w-sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
    realRoots.push_back(temp.real());
    temp = -B/(4.0*A) + 0.5*(-w+sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
    realRoots.push_back(temp.real());
    temp = -B/(4.0*A) + 0.5*(-w-sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
    realRoots.push_back(temp.real());
    
    return realRoots;
}

void p3p_kneip_main(
                                               const std::vector<Eigen::Vector3d> & f, //bearingVector_t
                                               const std::vector<Eigen::Vector3d> & p,         //points_t
                                               std::vector<Eigen::Matrix<double,3,4> > & solutions )
{
    Eigen::Vector3d P1 = p[0];
    Eigen::Vector3d P2 = p[1];
    Eigen::Vector3d P3 = p[2];
    
    Eigen::Vector3d temp1 = P2 - P1;
    Eigen::Vector3d temp2 = P3 - P1;
    
    if( temp1.cross(temp2).norm() == 0)
        return;
    
    Eigen::Vector3d f1 = f[0];
    Eigen::Vector3d f2 = f[1];
    Eigen::Vector3d f3 = f[2];
    
    Eigen::Vector3d e1 = f1;
    Eigen::Vector3d e3 = f1.cross(f2);
    e3 = e3/e3.norm();
    Eigen::Vector3d e2 = e3.cross(e1);
    
    Eigen::Matrix<double,3,3> T;
    T.row(0) = e1.transpose();
    T.row(1) = e2.transpose();
    T.row(2) = e3.transpose();
    
    f3 = T*f3;
    
    if( f3(2,0) > 0)
    {
        f1 = f[1];
        f2 = f[0];
        f3 = f[2];
        
        e1 = f1;
        e3 = f1.cross(f2);
        e3 = e3/e3.norm();
        e2 = e3.cross(e1);
        
        T.row(0) = e1.transpose();
        T.row(1) = e2.transpose();
        T.row(2) = e3.transpose();
        
        f3 = T*f3;
        
        P1 = p[1];
        P2 = p[0];
        P3 = p[2];
    }
    
    Eigen::Vector3d n1 = P2-P1;
    n1 = n1/n1.norm();
    Eigen::Vector3d n3 = n1.cross(P3-P1);
    n3 = n3/n3.norm();
    Eigen::Vector3d n2 = n3.cross(n1);
    
    Eigen::Matrix<double,3,3> N;
    N.row(0) = n1.transpose();
    N.row(1) = n2.transpose();
    N.row(2) = n3.transpose();
    
    P3 = N*(P3-P1);
    
    double d_12 = temp1.norm();
    double f_1 = f3(0,0)/f3(2,0);
    double f_2 = f3(1,0)/f3(2,0);
    double p_1 = P3(0,0);
    double p_2 = P3(1,0);
    
    double cos_beta = f1.dot(f2);
    double b = 1/( 1 - pow( cos_beta, 2 ) ) - 1;
    
    if( cos_beta < 0 )
        b = -sqrt(b);
    else
        b = sqrt(b);
    
    double f_1_pw2 = pow(f_1,2);
    double f_2_pw2 = pow(f_2,2);
    double p_1_pw2 = pow(p_1,2);
    double p_1_pw3 = p_1_pw2 * p_1;
    double p_1_pw4 = p_1_pw3 * p_1;
    double p_2_pw2 = pow(p_2,2);
    double p_2_pw3 = p_2_pw2 * p_2;
    double p_2_pw4 = p_2_pw3 * p_2;
    double d_12_pw2 = pow(d_12,2);
    double b_pw2 = pow(b,2);
    
    Eigen::Matrix<double,5,1> factors;
    
    factors(0,0) = -f_2_pw2*p_2_pw4
    -p_2_pw4*f_1_pw2
    -p_2_pw4;
    
    factors(1,0) = 2*p_2_pw3*d_12*b
    +2*f_2_pw2*p_2_pw3*d_12*b
    -2*f_2*p_2_pw3*f_1*d_12;
    
    factors(2,0) = -f_2_pw2*p_2_pw2*p_1_pw2
    -f_2_pw2*p_2_pw2*d_12_pw2*b_pw2
    -f_2_pw2*p_2_pw2*d_12_pw2
    +f_2_pw2*p_2_pw4
    +p_2_pw4*f_1_pw2
    +2*p_1*p_2_pw2*d_12
    +2*f_1*f_2*p_1*p_2_pw2*d_12*b
    -p_2_pw2*p_1_pw2*f_1_pw2
    +2*p_1*p_2_pw2*f_2_pw2*d_12
    -p_2_pw2*d_12_pw2*b_pw2
    -2*p_1_pw2*p_2_pw2;
    
    factors(3,0) = 2*p_1_pw2*p_2*d_12*b
    +2*f_2*p_2_pw3*f_1*d_12
    -2*f_2_pw2*p_2_pw3*d_12*b
    -2*p_1*p_2*d_12_pw2*b;
    
    factors(4,0) = -2*f_2*p_2_pw2*f_1*p_1*d_12*b
    +f_2_pw2*p_2_pw2*d_12_pw2
    +2*p_1_pw3*d_12
    -p_1_pw2*d_12_pw2
    +f_2_pw2*p_2_pw2*p_1_pw2
    -p_1_pw4
    -2*f_2_pw2*p_2_pw2*p_1*d_12
    +p_2_pw2*f_1_pw2*p_1_pw2
    +f_2_pw2*p_2_pw2*d_12_pw2*b_pw2;
    
    std::vector<double> realRoots = o4_roots(factors);
    
    for( int i = 0; i < 4; i++ )
    {
        double cot_alpha =
        (-f_1*p_1/f_2-realRoots[i]*p_2+d_12*b)/
        (-f_1*realRoots[i]*p_2/f_2+p_1-d_12);
        
        double cos_theta = realRoots[i];
        double sin_theta = sqrt(1-pow(realRoots[i],2));
        double sin_alpha = sqrt(1/(pow(cot_alpha,2)+1));
        double cos_alpha = sqrt(1-pow(sin_alpha,2));
        
        if (cot_alpha < 0)
            cos_alpha = -cos_alpha;
        
        Eigen::Matrix<double,3,1> C;
        C(0,0) = d_12*cos_alpha*(sin_alpha*b+cos_alpha);
        C(1,0) = cos_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha);
        C(2,0) = sin_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha);
        
        C = P1 + N.transpose()*C;
        
        Eigen::Matrix<double,3,3> R;
        R(0,0) = -cos_alpha;
        R(0,1) = -sin_alpha*cos_theta;
        R(0,2) = -sin_alpha*sin_theta;
        R(1,0) = sin_alpha;
        R(1,1) = -cos_alpha*cos_theta;
        R(1,2) = -cos_alpha*sin_theta;
        R(2,0) = 0.0;
        R(2,1) = -sin_theta;
        R(2,2) = cos_theta;
        
        R = N.transpose()*R.transpose()*T;
        
        Eigen::Matrix<double,3,4> solution;
        solution.col(3) = C;
        solution.block<3,3>(0,0) = R;
        
        solutions.push_back(solution);
    }
}

vector<cv::Affine3d> computeP3P(const std::vector<cv::Point2f> &projected2DPoints,const std::vector<cv::Point3f> &Model3DPoints)
{
    //convert to Eigen vectors
    std::vector<Eigen::Vector3d> points3d;
    for(unsigned int i(0); i < Model3DPoints.size(); ++i)
    {
        Eigen::Vector3d EigenVec(Model3DPoints[i].x,Model3DPoints[i].y,Model3DPoints[i].z);
        points3d.push_back(EigenVec);
    }

    std::vector<Eigen::Vector3d> bearingVectors;
    for (auto pp : projected2DPoints)
    {
        Eigen::Vector3d temp = Eigen::Vector3d(pp.x, pp.y, 1.);
        temp = temp/temp.norm();
        bearingVectors.push_back(temp);
    }

    std::vector<Eigen::Matrix<double,3,4> > poses;
    p3p_kneip_main(bearingVectors,points3d,poses);
    //for(int i(0); i < poses.size(); ++i)
    //    cout << " @@@@@@@@   P3P pose [R | t]  no. " << i << endl<< poses[i]<<endl<<endl;

    //convert Eigen poses to openCV poses
    vector<cv::Affine3d> poses_cv;
    for(unsigned int i(0); i < poses.size(); ++i)
    {
        Eigen::Matrix<double,3,4> &pose_c = poses[i];
        Eigen::AngleAxis<double> axisRot(pose_c.block<3,3>(0,0));
        Eigen::Vector3d r = axisRot.angle()*axisRot.axis();

        cv::Vec3d r_cv(r[0],r[1],r[2]);
        cv::Vec3d t_cv(pose_c(0,3),pose_c(1,3),pose_c(2,3));

        cv::Affine3d pose_cv(r_cv,t_cv);
        //cout << " @@@@@@@@   P3P pose [R | t]  no. " << i << endl<<pose_cv.matrix<<endl<<endl;

        poses_cv.push_back(pose_cv);

    }
    return poses_cv;
    

}

}
