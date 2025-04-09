#include <opencv2/opencv.hpp>
#include <iostream>

void resizeImage(const std::string& inputImagePath, const std::string& outputImagePath) {
    // 读取原始图像
    cv::Mat image = cv::imread(inputImagePath);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return;
    }

    // 检查图像是否是 640x640
    if (image.rows != 640 || image.cols != 640) {
        std::cerr << "Error: The input image is not 640x640." << std::endl;
        return;
    }

    // 创建一个新的 Mat 对象，用于存储调整大小后的图像
    cv::Mat resizedImage;

    // 调整图像大小为 640x480
    cv::resize(image, resizedImage, cv::Size(640, 480));

    // 保存调整大小后的图像
    if (!cv::imwrite(outputImagePath, resizedImage)) {
        std::cerr << "Error: Unable to save resized image!" << std::endl;
        return;
    }

    std::cout << "Resized image saved to: " << outputImagePath << std::endl;
}

int main() {
    // 输入图像路径和输出图像路径
    std::string inputImagePath = "input_image.jpg"; // 替换为实际的输入图像路径
    std::string outputImagePath = "resized_image.jpg"; // 替换为输出图像路径

    // 调用 resizeImage 函数
    resizeImage(inputImagePath, outputImagePath);

    return 0;
}

