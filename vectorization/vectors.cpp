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