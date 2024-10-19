#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::vector<float> extractEmbedding(const std::string& imagePath) {
    // Load the image using OpenCV
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    cv::resize(image, image, cv::Size(256, 256));
    int cropSize = 224;
    int offsetX = (image.cols - cropSize) / 2;
    int offsetY = (image.rows - cropSize) / 2;
    cv::Rect cropRegion(offsetX, offsetY, cropSize, cropSize);
    image = image(cropRegion);

    // Convert to float and normalize
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(image, mean, image);
    cv::divide(image, std, image);