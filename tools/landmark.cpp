//usage example :
//./landmark ../data/landmarks/marker.png ../data/landmarks/marker.xml.gz
//./landmark ../data/landmarks/ziggu.png ../data/landmarks/ziggu.xml.gz

#include <iostream>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

static constexpr float pi = 3.141592654;

#define DEBUG 1

namespace detail
{
    static constexpr float fov = 50.f * 2 * pi / 360.f;
    static const float f = 1.f / (2 * std::tan(fov / 2));
}

static const cv::Mat idealCalibrationMatrix = (cv::Mat_<float>(3, 4) <<
    800*detail::f, 0, 400, 0,
    0, 800*detail::f, 300, 0,
    0, 0, 1, 0);

typedef std::pair<float, float> Orientation;

cv::Mat transform_matrix(const Orientation& orientation, float t_z)
{
    float pitch = orientation.first;
    float roll = orientation.second;
    
    const cv::Mat roll_matrix = (cv::Mat_<float>(4, 4) <<
        std::cos(roll), std::sin(roll), 0, 0,
        -std::sin(roll), std::cos(roll), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
    
    const cv::Mat pitch_matrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 0, 0,
        0, std::cos(pitch), std::sin(pitch), 0,
        0, -std::sin(pitch), std::cos(pitch), 0,
        0, 0, 0, 1);
    
    cv::Mat res = pitch_matrix * roll_matrix;
    res.at<float>(2, 3) = t_z;
    return res;
}

cv::Point2f point_max(const cv::Point2f& p1, const cv::Point2f& p2)
{
    auto max_x = std::max(p1.x, p2.x);
    auto max_y = std::max(p1.y, p2.y);
    return cv::Point2f(max_x, max_y);
}

cv::Point2f point_min(const cv::Point2f& p1, const cv::Point2f& p2)
{
    auto min_x = std::min(p1.x, p2.x);
    auto min_y = std::min(p1.y, p2.y);
    return cv::Point2f(min_x, min_y);
}

void tightFrame(const std::vector<cv::Point2f>& points, float margin,
                std::vector<cv::Point2f>& newPoints, cv::Size& newSize)
{
    cv::Point2f max = std::accumulate(points.begin(), points.end(), cv::Point2f(0, 0), point_max);
    cv::Point2f min = std::accumulate(points.begin(), points.end(), max, point_min);
    
    // Include margin
    min -= cv::Point2f(margin, margin);
    
    // Transform points with new margin
    newPoints.resize(points.size());
    std::transform(points.begin(), points.end(), newPoints.begin(),
        [&](const cv::Point2f& p){return p - min;}
    );
    
    // Compute new image size
    cv::Point2f aux = max - min + cv::Point2f(margin, margin);
    newSize = cv::Size(std::ceil(aux.x), std::ceil(aux.y));
}

cv::Mat getHomography(const Orientation& orientation,
                      const cv::Size& imageSize,
                      const cv::Size2f& realSize,
                      const cv::Mat& calibrationMatrix,
                      int margin,
                      cv::Size& newSize)
{
    static const std::vector<cv::Point3f> template_points =
        {
            {-0.5, -0.5, 0},
            {0.5, -0.5, 0},
            {0.5, 0.5, 0},
            {-0.5, 0.5, 0}
        };
    
    // Get template corners in real world coordinates
    std::vector<cv::Point3f> real_points;
    for(auto p : template_points)
    {
        cv::Point3f q = {p.x * realSize.width, p.y * realSize.height, p.z};
        real_points.push_back(q);
    }
    
    // Transform the template for the given orientation
    
    // Get the transformation matrix
    float t_z = 5.f * std::max(realSize.width, realSize.height);
    const cv::Mat m = transform_matrix(orientation, t_z);
    
    // Transform and project the points
    std::vector<cv::Point3f> transfPoints;
    cv::transform(real_points, transfPoints, calibrationMatrix * m);
    
    std::vector<cv::Point2f> projPoints(4);
    cv::convertPointsFromHomogeneous(transfPoints, projPoints);
    
    // Add margin and compute image size after homography
    std::vector<cv::Point2f> newPoints;
    tightFrame(projPoints, margin, newPoints, newSize);
    
    // Compute the homography
    cv::Point2f aux(imageSize.width, imageSize.height);
    const std::vector<cv::Point2f> imagePoints =
        {
            {0, 0},
            {aux.x, 0},
            {aux.x, aux.y},
            {0, aux.y}
        };
    
    cv::Mat h = cv::getPerspectiveTransform(imagePoints, newPoints);
    
    return h;
}

void projectKeypoints(const cv::Mat& homography, std::vector<cv::KeyPoint>& keypoints)
{
    std::vector<cv::Point2f> points, newPoints;
    for(auto kp : keypoints)
        points.push_back(kp.pt);
    
    cv::perspectiveTransform(points, newPoints, homography);
    
    auto newPointIt = newPoints.begin();
    auto keypointIt = keypoints.begin();
    for(; keypointIt != keypoints.end(); ++keypointIt, ++newPointIt)
        keypointIt->pt = *newPointIt;
}

class LandmarkProcessor
{
public:
    LandmarkProcessor(float pixelsPerUnit,
                      cv::Ptr<cv::Feature2D> featureExtractor)
        : mPixelsPerUnit(pixelsPerUnit)
        , mFeatureExtractor(featureExtractor)
        , mCalibrationMatrix(idealCalibrationMatrix)
    {}
    
    void process(const cv::Mat& image,
                 const std::vector<Orientation>& orientations,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors,
                 cv::Size2f& realSize) const;
    
    inline const cv::Mat& getCalibrationMatrix() const {return mCalibrationMatrix;}

private:
    float mPixelsPerUnit;
    cv::Ptr<cv::Feature2D> mFeatureExtractor;
    cv::Mat mCalibrationMatrix;
};

void LandmarkProcessor::process(const cv::Mat& image,
                                const std::vector<Orientation>& orientations,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors,
                                cv::Size2f& realSize) const
{
    cv::Size imageSize = image.size();
    realSize = cv::Size(imageSize.width / mPixelsPerUnit, imageSize.height / mPixelsPerUnit);
    
    std::vector<cv::Mat> descriptorsVector;
    keypoints.clear();
    
    for(auto orientation : orientations)
    {
        cv::Size newSize;
        cv::Mat h = getHomography(orientation, imageSize, realSize, mCalibrationMatrix, 50, newSize);
        
        cv::Mat image2;
        cv::warpPerspective(image, image2, h, newSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        
        std::vector<cv::KeyPoint> cur_keypoints;
        cv::Mat cur_descriptors;
        mFeatureExtractor->detectAndCompute(image2, cv::noArray(), cur_keypoints, cur_descriptors);
        std::cout<<"Nb features : "<<cur_keypoints.size()<<std::endl;
        
#if DEBUG
        cv::Mat image3 = image2;
        cv::drawKeypoints(image2, cur_keypoints, image3);
        cv::imwrite("debug.png", image3);
#endif
        
        projectKeypoints(h.inv(), cur_keypoints);
        
        keypoints.insert(keypoints.end(), cur_keypoints.begin(), cur_keypoints.end());
        descriptorsVector.push_back(cur_descriptors);
    }
    
    cv::vconcat(descriptorsVector, descriptors);
}

void print_usage(const char* command)
{
    std::cerr << "Usage:\n\t" << command << " <image file> <output file> [<pixels per unit>]" << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc < 3 || argc > 4)
    {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string imageFilename = argv[1];
    std::string outputFilename = argv[2];
    float pixelsPerUnit = 30.0;
    
    if(argc >= 4)
        pixelsPerUnit = std::stof(argv[3]);
    
    cv::Ptr<cv::Feature2D> fextractor = cv::BRISK::create(60);
    //cv::Ptr<cv::Feature2D> fextractor = cv::BRISK::create(30);
    // cv::Ptr<cv::Feature2D> fextractor = new brisk::BriskFeature(10.0, 4);
    LandmarkProcessor landmarkProcessor(pixelsPerUnit, fextractor);
    
    // Load image
    cv::Mat image = cv::imread(imageFilename);
    cv::Mat originalImage; image.copyTo(originalImage);

    if(image.empty())
    {
        std::cerr << "Could not open " << imageFilename << ". " << std::endl;
        return 1;
    }
    
    if(image.type() != CV_8UC1)
    {
        cv::cvtColor(image, image, CV_RGB2GRAY);
    }
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Size2f realSize;
    static const std::vector<Orientation> orientations =
        {
            {0, 0},
            // {4 * pi / 12, 0}
        };
    landmarkProcessor.process(image,
                              orientations,
                              keypoints,
                              descriptors,
                              realSize);
    
    // Save
    cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    cv::write(fs, "keypoints", keypoints);
    cv::write(fs, "descriptors", descriptors);
    cv::write(fs, "image_size", image.size());
    cv::write(fs, "real_size", realSize);

    //BRISK and KLT work only on grayscale image 
    //active search works for color images => check whether
    //color is beneficial considering timing trade off
    cv::write(fs, "image", image);
    //cv::write(fs, "image", originalImage);
    fs.release();
    
    return 0;
}
