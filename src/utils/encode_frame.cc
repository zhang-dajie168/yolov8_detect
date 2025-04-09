#include <iostream>
#include <vector>
#include <turbojpeg.h>
#include <opencv2/opencv.hpp>
#include <common.h>


static const char* subsampName[TJ_NUMSAMP] = {"4:4:4", "4:2:2", "4:2:0", "Grayscale", "4:4:0", "4:1:1"};
static const char* colorspaceName[TJ_NUMCS] = {"RGB", "YCbCr", "GRAY", "CMYK", "YCCK"};



// OpenCV Mat 编码为 JPEG（使用 TurboJPEG）
std::vector<uchar> tjpeg_encode(const cv::Mat& image, int quality = 95) {
    if (image.empty() || image.type() != CV_8UC3) {
        std::cerr << "Error: 输入图像必须为 BGR 格式 (CV_8UC3)" << std::endl;
        return {};
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    tjhandle handle = tjInitCompress();
    if (!handle) {
        std::cerr << "Error: tjInitCompress failed" << std::endl;
        return {};
    }

    // 压缩参数配置
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    const int pixel_format = TJPF_RGB; // 输入图像格式
    const int subsamp = TJSAMP_420;    // 子采样方式（4:2:0 最高质量）

    // 执行压缩
    if (tjCompress2(handle, image.data, image.cols, 0, image.rows, pixel_format,
                   &jpeg_buf, &jpeg_size, subsamp, quality, TJFLAG_FASTDCT) != 0) {
        std::cerr << "Error: tjCompress2 failed - " << tjGetErrorStr2(handle) << std::endl;
        tjDestroy(handle);
        return {};
    }

    // 拷贝数据到 vector
    //std::vector<uchar> jpeg_data(jpeg_buf, jpeg_size);
    std::vector<uchar> jpeg_data(jpeg_buf, jpeg_buf + jpeg_size);
    tjFree(jpeg_buf); // 必须释放 TurboJPEG 分配的内存
    tjDestroy(handle);
    return jpeg_data;
}


// OpenCV Mat 编码为 JPEG（使用 TurboJPEG）
int  tjpeg_encode_frame(const cv::Mat& frame, image_buffer_t* image,int quality = 95) {

    if (frame.empty() || frame.type() != CV_8UC3) {
        std::cerr << "Error: 输入图像必须为 BGR 格式 (CV_8UC3)" << std::endl;
        return {};
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    unsigned long jpegSize;
    int flags = 0;
    int width, height;
    int origin_width, origin_height;
    unsigned char* imgBuf = NULL;
    unsigned char* jpegBuf = NULL;
    unsigned long size;
    unsigned short orientation = 1;
    struct timeval tv1, tv2;
    int pixelFormat = TJPF_RGB;


    const int pixel_format = TJPF_RGB; // 输入图像格式
    const int subsamp = TJSAMP_420;    // 子采样方式（4:2:0 最高质量）

    // 执行压缩
    tjhandle handle = tjInitCompress();
    if (!handle) {
        std::cerr << "Error: tjInitCompress failed" << std::endl;
        return {};
    }

    if (tjCompress2(handle, frame.data, frame.cols, 0, frame.rows, pixel_format,
                   &jpegBuf, &size, subsamp, quality, TJFLAG_FASTDCT) != 0) {
        std::cerr << "Error: tjCompress2 failed - " << tjGetErrorStr2(handle) << std::endl;
        tjDestroy(handle);
        return {};
    }

    if (size == 0) {
        printf("determining input file size, Input file contains no data\n");
    }
    jpegSize = (unsigned long)size;
    if ((jpegBuf = (unsigned char*)malloc(jpegSize * sizeof(unsigned char))) == NULL) {
        printf("allocating JPEG buffer\n");
    }
    //解码
    //tjhandle handle = NULL;
    int subsample, colorspace;
    int padding = 1;
    int ret = 0;

    handle = tjInitDecompress();
    ret = tjDecompressHeader3(handle, jpegBuf, size, &origin_width, &origin_height, &subsample, &colorspace);
    if (ret < 0) {
        printf("header file error, errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
        return -1;
    }

    // 对图像做裁剪16对齐，利于后续rga操作
    int crop_width = origin_width / 16 * 16;
    int crop_height = origin_height / 16 * 16;

    printf("origin size=%dx%d crop size=%dx%d\n", origin_width, origin_height, crop_width, crop_height);

    // gettimeofday(&tv1, NULL);
    ret = tjDecompressHeader3(handle, jpegBuf, size, &width, &height, &subsample, &colorspace);
    if (ret < 0) {
        printf("header file error, errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
        return -1;
    }
    printf("input image: %d x %d, subsampling: %s, colorspace: %s, orientation: %d\n", 
            width, height, subsampName[subsample], colorspaceName[colorspace], orientation);
    int sw_out_size = width * height * 3;
    unsigned char* sw_out_buf = image->virt_addr;
    if (sw_out_buf == NULL) {
        sw_out_buf = (unsigned char*)malloc(sw_out_size * sizeof(unsigned char));
    }
    if (sw_out_buf == NULL) {
        printf("sw_out_buf is NULL\n");
        goto out;
    }

    flags |= 0;

    // 错误码为0时，表示警告，错误码为-1时表示错误

    ret = tjDecompress2(handle, jpegBuf, size, sw_out_buf, width, 0, height, pixelFormat, flags);
    // ret = tjDecompressToYUV2(handle, jpeg_buf, size, dst_buf, *width, padding, *height, flags);
    if ((0 != tjGetErrorCode(handle)) && (ret < 0)) {
        printf("error : decompress to yuv failed, errorStr:%s, errorCode:%d\n", tjGetErrorStr(),
               tjGetErrorCode(handle));
        goto out;
    }
    if ((0 == tjGetErrorCode(handle)) && (ret < 0)) {
        printf("warning : errorStr:%s, errorCode:%d\n", tjGetErrorStr(), tjGetErrorCode(handle));
    }
    tjDestroy(handle);
    // gettimeofday(&tv2, NULL);
    // printf("decode time %ld ms\n", (tv2.tv_sec-tv1.tv_sec)*1000 + (tv2.tv_usec-tv1.tv_usec)/1000);

    image->width = width;
    image->height = height;
    image->format = IMAGE_FORMAT_RGB888;
    image->virt_addr = sw_out_buf;
    image->size = sw_out_size;
out:
    if (jpegBuf) {
        free(jpegBuf);
    }
    return 0;
}

