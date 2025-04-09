/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <opencv2/opencv.hpp>
#include <common.h>
#include <iostream>
#include <turbojpeg.h>
#include <vector>
#include "encode_frame.h"


#include "bytetrack/BYTETracker.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/polygon_stamped.hpp"

#include <chrono>
#include <image_transport/image_transport.hpp>
#include <iostream>

class Yolov8TrackerNode : public rclcpp::Node
{
public:
    Yolov8TrackerNode() : Node("yolov8_tracker_node")
    {
        // 获取参数
        this->declare_parameter<std::string>("model_path", "./yolov8.rknn");
        
        std::string model_path_ = this->get_parameter("model_path").as_string();

        // 初始化 YOLO 模型
        int ret;
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        init_post_process();

        if(init_yolov8_model(model_path_.c_str(), &rknn_app_ctx) != 0){
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize YOLOv8 model!");
            rclcpp::shutdown();
        }

        // 初始化订阅器，订阅图像话题
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>
        (
            "/camera/color/image_raw", 
            rclcpp::SensorDataQoS().keep_last(30),
            [this, &rknn_app_ctx](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                this->image_callback(msg, &rknn_app_ctx);}  
         );

         // 初始化发布图像话题
        image_pub_=this->create_publisher<sensor_msgs::msg::Image>("output_image",30);
        
        // 初始化发布跟踪话题
        tracked_pub_=this->create_publisher<geometry_msgs::msg::PolygonStamped>("/tracked_objects",30);
        
        //初始化 跟踪器
        tracker_=std::make_unique<BYTETracker>(30, 30);
    }


private:

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg, rknn_app_context_t* rknn_app_ctx)
    {
        try {
            // // 转换ROS图像到OpenCV格式
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            
            // 初始化准备输入图像
            image_buffer_t src_image;
            memset(&src_image, 0, sizeof(image_buffer_t));


            std::vector<uchar> jpegData;
            unsigned char* jpegBuf = nullptr;
            unsigned long jpegSize = 0;

            // 转换到图像缓冲区
            jpegData.clear();    

            jpegData = tjpeg_encode(frame, 95);// 使用 TurboJPEG 编码为 JPEG
            jpegBuf = jpegData.data(); // 获取JPEG数据的指针
            jpegSize = jpegData.size();  // 获取JPEG数据的大小

            // 检查 JPEG 数据是否有效
            if (jpegBuf == nullptr || jpegSize == 0) {
                std::cerr << "JPEG encoding failed or returned empty data!" << std::endl;
                return;
            }
  
            read_frame_jpeg(jpegBuf,jpegSize,&src_image) ;  //Mat::frame->image_buffer_t src_image

            // 执行目标检测
            object_detect_result_list od_results;


            if(inference_yolov8_model(rknn_app_ctx, &src_image, &od_results) != 0){
                RCLCPP_ERROR(this->get_logger(), "Inference failed!");
                return;
            }
  
            // 转换检测结果到跟踪格式
            std::vector<Object> trackobj;
            publish_decobj_to_trackobj(od_results, trackobj);
         
            // 更新跟踪器
            auto tracks = tracker_->update(trackobj);

            // 绘制结果
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            draw_tracking_results(frame, tracks);
            
            // 发布处理后的图像
            sensor_msgs::msg::Image::SharedPtr output_msg=cv_bridge::CvImage(msg->header,"bgr8",frame).toImageMsg();
            image_pub_->publish(*output_msg);
            
            if (src_image.virt_addr != NULL){
            	free(src_image.virt_addr);
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        
            deinit_post_process();

            int ret = release_yolov8_model(rknn_app_ctx);
            if (ret != 0){
            printf("release_yolov8_model fail! ret=%d\n", ret);
            }   
        }
    }

    void publish_decobj_to_trackobj(object_detect_result_list &results, std::vector<Object> &trackobj)
    {
        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp=this->get_clock()->now();
        polygon_msg.header.frame_id="camera_link";
        
        if (results.count >0)
        {
            trackobj.clear();
        }
    	for (int i = 0; i < results.count; i++) {
            Object obj;
            obj.classId = results.results[i].cls_id;
            obj.score = results.results[i].prop;
            obj.box = cv::Rect(
                    results.results[i].box.left,
                    results.results[i].box.top,
                    results.results[i].box.right - results.results[i].box.left,
                    results.results[i].box.bottom - results.results[i].box.top);
                

            trackobj.push_back(obj);
        }
        
        tracked_pub_->publish(polygon_msg);
    }

    void draw_tracking_results(cv::Mat &img, const std::vector<STrack> &tracks)
    {
        std::cout<<"Tracking result (output_stracks):"<<std::endl;
        for (const auto &track : tracks)
        {
            // 画出跟踪框
            std::cout<<"track_id: "<<track.track_id<<",Track  Bounding Box:["<<track.tlbr[0]<<","<<track.tlbr[1]<<","<<track.tlbr[2]<<","<<track.tlbr[3]<<"]"<<",sorce:"<<track.score<<std::endl;
            int x1=static_cast<int>(track.tlbr[0]);
            int y1=static_cast<int>(track.tlbr[1]);
            int x2=static_cast<int>(track.tlbr[2]);
            int y2=static_cast<int>(track.tlbr[3]);
            cv::Rect rect(x1,y1,x2-x1,y2-y1);
            cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

            // 显示跟踪 ID
            int text_x=x1;
            int text_y=y1-5;
            
            cv::putText(img, std::to_string(track.track_id), cv::Point(text_x,text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }     
    }   
        //声明对象
    
    std::unique_ptr<BYTETracker>tracker_;
    std::vector<Object> track_objs;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr tracked_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;


};


int main(int argc, char **argv)
{
    rclcpp::InitOptions init_options;
    init_options.shutdown_on_signal=true; 
    rclcpp::init(argc, argv,init_options);
    rclcpp::spin(std::make_shared<Yolov8TrackerNode>());
    rclcpp::shutdown();
    return 0;
}

