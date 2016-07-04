
#include "VideoSource.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

//static const char window_name[] = "Landmarks";

//namespace tt = thymio_tracker;

static constexpr float pi = 3.141592654;

namespace detail
{
    static constexpr float fov = 50.f * 2 * pi / 360.f;
    static const float f = 1.f / (2 * std::tan(fov / 2));
}


int main(int argc, char** argv)
{
    float ppu = 30.;
    cv::Ptr<cv::BRISK> test_orb = cv::BRISK::create(60);//(int thresh=30, int octaves=3, float patternScale=1.0f)
    
    std::vector<cv::KeyPoint> templateKeypoints;
    cv::Mat templateDescriptors;
    
    cv::Mat templateImage = cv::imread("../data/landmarks/marker_noborder.png");

    //now resize the template.
    //let s say that we resize it to the size it would be if it was seen from a distance equal 5x its width
    //float real_width = templateImage.size().width / ppu;
    float new_width = 800*detail::f / 5.;
    float scale = new_width / templateImage.size().width;

    cv::Mat templateImageRescaled;
    cv::resize(templateImage,templateImageRescaled,cv::Size(scale*templateImage.size().width,scale*templateImage.size().height));

    //add margin
    int margin = 50;
    cv::Mat templateImageBorder;
    copyMakeBorder( templateImageRescaled, templateImageBorder, margin, margin, margin, margin, cv::BORDER_CONSTANT, cv::Scalar(255,255,255) );
    cv::imwrite("../data/landmarks/marker_noborder_rescaled.png",templateImageBorder);
    
    //test_orb->detectAndCompute(templateImage, cv::noArray(), templateKeypoints, templateDescriptors);
    test_orb->detectAndCompute(templateImageBorder, cv::noArray(), templateKeypoints, templateDescriptors);
    std::cout<<"Nb features : "<<templateKeypoints.size()<<std::endl;

    //rescale template Keypoint positions
    for(unsigned int i=0;i<templateKeypoints.size();i++)
        templateKeypoints[i].pt = (templateKeypoints[i].pt-cv::Point2f(margin,margin))/scale;


    cv::FileStorage fs("../data/landmarks/markerTest.xml.gz", cv::FileStorage::WRITE);
    cv::write(fs, "image", templateImage);
    cv::write(fs, "image_size", templateImage.size());
    cv::write(fs, "real_size", cv::Size2f(templateImage.size())/ppu);
    cv::write(fs,"keypoints", templateKeypoints);
    cv::write(fs,"descriptors", templateDescriptors);
    fs.release();
    
    return 0;

}
/*
int main(int argc, char** argv)
{
    VideoSourceLive videoSource(EmbeddedCam);
    videoSource.resizeSource(0.5);
    
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::Mat outputImage;
    
    // cv::Ptr<cv::ORB> train_orb = cv::ORB::create(500, 1.2f, 8);
    cv::Ptr<cv::ORB> test_orb = cv::ORB::create(500);
    // cv::Ptr<cv::xfeatures2d::FREAK> test_orb = cv::xfeatures2d::FREAK::create();
    // cv::Ptr<cv::BRISK> test_orb = cv::BRISK::create();
    
    std::vector<cv::KeyPoint> detectedKeypoints;
    std::vector<cv::KeyPoint> templateKeypoints;
    cv::Mat detectedDescriptors;
    cv::Mat templateDescriptors;
    
    cv::Mat templateImage = cv::imread("../data/landmarks/oack.png");
    // cv::Mat templateImage = cv::imread("../data/landmarks/hut.png");
    
    cv::FileStorage fs("oack.xml", cv::FileStorage::READ);
    cv::read(fs["keypoints"], templateKeypoints);
    cv::read(fs["descriptors"], templateDescriptors);
    fs.release();
    
    // cv::FileStorage fs2("test.xml", cv::FileStorage::WRITE);
    // cv::write(fs2, "image_size", cv::Size2i(400, 300));
    // cv::write(fs2, "real_size", cv::Size2f(34.5, 20.4));
    // fs2.release();
    
    // return 0;
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch> > matches;
    // matcher.add(templateDescriptors);
    
    while(true)
    {
        videoSource.grabNewFrame();
        const cv::Mat& inputImage = videoSource.getFramePointer();
        
        test_orb->detectAndCompute(inputImage, cv::noArray(), detectedKeypoints, detectedDescriptors);
        
        matches.clear();
        matcher.knnMatch(detectedDescriptors, templateDescriptors, matches, 2);
        
        std::vector<cv::DMatch> goodMatches;
        for(auto el : matches)
        {
            if(el[0].distance < 0.75 * el[1].distance)
                goodMatches.push_back(el[0]);
        }
        
        // Fit a homography
        std::vector<cv::Point2f> object;
        std::vector<cv::Point2f> scene;
        for(auto match : goodMatches)
        {
            object.push_back(templateKeypoints[match.trainIdx].pt);
            scene.push_back(detectedKeypoints[match.queryIdx].pt);
        }
        
        cv::Mat h;
        if(!scene.empty())
            h = cv::findHomography(object, scene, CV_RANSAC);
        
        if(!h.empty())
        {
            std::vector<cv::Point2f> obj_corners(4);
            obj_corners[0] = cv::Point2f(0, 0);
            obj_corners[1] = cv::Point2f(templateImage.cols, 0);
            obj_corners[2] = cv::Point2f(templateImage.cols, templateImage.rows);
            obj_corners[3] = cv::Point2f(0, templateImage.rows);
            
            std::vector<cv::Point2f> scene_corners(4);
            cv::perspectiveTransform(obj_corners, scene_corners, h);
            
            cv::line(inputImage, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 2);
            cv::line(inputImage, scene_corners[1], scene_corners[2], cv::Scalar(0, 255, 0), 2);
            cv::line(inputImage, scene_corners[2], scene_corners[3], cv::Scalar(0, 255, 0), 2);
            cv::line(inputImage, scene_corners[3], scene_corners[0], cv::Scalar(0, 255, 0), 2);
        }
        
        inputImage.copyTo(outputImage);
        cv::drawKeypoints(inputImage, detectedKeypoints, outputImage);
        // cv::drawMatches(inputImage, detectedKeypoints,
        //                 templateImage, templateKeypoints,
        //                 goodMatches, outputImage,
        //                 cv::Scalar(0, 0, 255),
        //                 cv::Scalar(255, 0, 0));
        
        imshow(window_name, outputImage);
        
        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q')
            break;
    }
    
    return 0;
}
*/