/*

create a descriptor from a landmark image, the output xml file will be loaded by ThimyoTracker
if it is added to the config file 

*/

#include "VideoSource.hpp"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


void print_usage(const char* command)
{
    std::cerr << "Usage :\n\t" << command << " <image file> <output file> <scale factor> <width in meters> <height in meters>" << std::endl;
    std::cerr << "example :\n\t./landmark_simple ../data/landmarks/marker.png  ../data/landmarks/marker2.xml.gz 0.5 0.1025 0.1025" << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc !=6)
    {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string imageFilename = argv[1];
    std::string outputFilename = argv[2];

    float scale = std::stof(argv[3]);
    float r_width = std::stof(argv[4]);
    float r_height = std::stof(argv[5]);


    cv::Mat templateImage = cv::imread(imageFilename);
    if(templateImage.type() != CV_8UC1)
        cv::cvtColor(templateImage, templateImage, CV_RGB2GRAY);

    //resize the template
    cv::Mat templateImageRescaled;
    //can resize usign cv::resize but if downsample then it will cause alisasing
    //cv::resize(templateImage,templateImageRescaled,cv::Size(scale*templateImage.size().width,scale*templateImage.size().height),cv::INTER_LINEAR);
    
    //create iamge pyramid for active search
    //compute number of levels
    int nbLevels = (int)(0.5-log(scale)/log(2)) + 1;
    //fill pyramid
    std::vector<cv::Mat> mPyramid;
    mPyramid.resize(nbLevels);
    templateImage.copyTo(mPyramid[0]);
    for(int l=1;l<nbLevels;l++)
        pyrDown( mPyramid[l-1], mPyramid[l], cv::Size( mPyramid[l-1].cols/2, mPyramid[l-1].rows/2 ) );
    std::cout<<"Lvl Pyramid used : "<<nbLevels-1<<std::endl;
    cv::resize(mPyramid[nbLevels-1],templateImageRescaled,cv::Size(scale*templateImage.size().width,scale*templateImage.size().height),cv::INTER_LINEAR);

    //add margin
    int margin = 50;
    cv::Mat templateImageBorder;
    copyMakeBorder( templateImageRescaled, templateImageBorder, margin, margin, margin, margin, cv::BORDER_CONSTANT, cv::Scalar(255,255,255) );
    
    //create feature extractor and extract
    cv::Ptr<cv::BRISK> test_orb = cv::BRISK::create(60);//(int thresh=30, int octaves=3, float patternScale=1.0f)
    std::vector<cv::KeyPoint> templateKeypoints;
    cv::Mat templateDescriptors;
    test_orb->detectAndCompute(templateImageBorder, cv::noArray(), templateKeypoints, templateDescriptors);
    std::cout<<"Nb features : "<<templateKeypoints.size()<<std::endl;

    //rescale template Keypoint positions to original image coordinate
    for(unsigned int i=0;i<templateKeypoints.size();i++)
        templateKeypoints[i].pt = (templateKeypoints[i].pt-cv::Point2f(margin,margin))/scale;

    //save to file
    cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    cv::write(fs, "image", templateImage);
    cv::write(fs, "image_size", templateImage.size());
    cv::write(fs, "real_size", cv::Size2f(r_width,r_height));
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