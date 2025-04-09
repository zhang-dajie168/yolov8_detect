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
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/polygon_stamped.hpp"

#include <chrono>
#include <image_transport/image_transport.hpp>
#include <iostream>
#include <sys/time.h>


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

        depth_image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 30, std::bind(&DepthConverter::depthCallback, this, std::placeholders::_1));

        camera_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 30, std::bind(&DepthConverter::cameraInfoCallback, this, std::placeholders::_1));

        bridge_ = std::make_shared<cv_bridge::CvBridge>();

         // 初始化内参
         fx_ = fy_ = cx_ = cy_ = 0;

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
            auto start_1 = std::chrono::high_resolution_clock::now();
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

            auto start_2 = std::chrono::high_resolution_clock::now();
            auto pre_time = std::chrono::duration_cast<std::chrono::microseconds>(start_2 - start_1).count() / 1000.0;
            std::cout<<"pre_time:"<<pre_time<<"ms"<<std::endl;
            
            if(inference_yolov8_model(rknn_app_ctx, &src_image, &od_results) != 0){
                RCLCPP_ERROR(this->get_logger(), "Inference failed!");
                return;
            }
            printf("yolo run success!!\n");
   
            // //打印检测结果
            // for(int i = 0; i < od_results.count; i++){
            //     Object obj;
            //     obj.box = cv::Rect(
            //         od_results.results[i].box.left,
            //         od_results.results[i].box.top,
            //         od_results.results[i].box.right - od_results.results[i].box.left,
            //         od_results.results[i].box.bottom - od_results.results[i].box.top);
                
            //     obj.score = od_results.results[i].prop;
            //     obj.classId = od_results.results[i].cls_id;
                
            //     std::cout<<" obj.classId:"<< obj.classId<<" obj.score:"<< obj.score<<" obj.box:"<< obj.box<<std::endl;
            // }

            auto start_3 = std::chrono::high_resolution_clock::now();
            auto yolo_time = std::chrono::duration_cast<std::chrono::microseconds>(start_3 - start_2).count() / 1000.0;
            std::cout<<"yolo_time:"<<yolo_time<<"ms"<<std::endl;
            
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
            
            	      // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            auto track_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start_3).count() / 1000.0;
            auto all_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start_1).count() / 1000.0;
            
            std::cout<<"track_time:"<<track_time<<"ms"<<std::endl; 
	        std::cout<<"all_time:"<<all_time<<"ms"<<std::endl;
         
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
    // 创建发布的消息类型
    tracked_objects_msgs::msg::TrackedObjectList tracked_object_list_msg;
    tracked_object_list_msg.header.stamp = this->get_clock()->now();
    tracked_object_list_msg.header.frame_id = "camera_link"; // 设置frame_id

    // 清空 trackobj
    trackobj.clear();
    
    if (results.count > 0)
    {
        // 遍历检测结果，构造 TrackedObject 消息并填充到 tracked_object_list_msg
        for (int i = 0; i < results.count; i++) {
            if (results.results[i].cls_id != 0) continue; // 过滤不需要的类别
            
            tracked_objects_msgs::msg::TrackedObject tracked_obj;
            tracked_obj.classId = results.results[i].cls_id;
            tracked_obj.score = results.results[i].prop;

            // 这里假设你想将边界框转换为多边形
            geometry_msgs::msg::Polygon polygon;
            geometry_msgs::msg::Point32 p1, p2, p3, p4;

            p1.x = results.results[i].box.left;
            p1.y = results.results[i].box.top;
            p2.x = results.results[i].box.right;
            p2.y = results.results[i].box.top;
            p3.x = results.results[i].box.right;
            p3.y = results.results[i].box.bottom;
            p4.x = results.results[i].box.left;
            p4.y = results.results[i].box.bottom;

            // 将四个顶点加入到多边形中
            polygon.points.push_back(p1);
            polygon.points.push_back(p2);
            polygon.points.push_back(p3);
            polygon.points.push_back(p4);

            tracked_obj.polygon = polygon;

            // 将当前物体添加到物体列表
            tracked_object_list_msg.objects.push_back(tracked_obj);

            // 将物体信息添加到 trackobj 向量中
            Object obj;
            obj.classId = tracked_obj.classId;
            obj.score = tracked_obj.score;
            obj.box = cv::Rect(
                    results.results[i].box.left,
                    results.results[i].box.top,
                    results.results[i].box.right - results.results[i].box.left,
                    results.results[i].box.bottom - results.results[i].box.top);

            trackobj.push_back(obj);
        }
    }

    // 发布消息
    tracked_pub_->publish(tracked_object_list_msg);
}


    void draw_tracking_results(cv::Mat &img, const std::vector<STrack> &tracks)
    {
        std::cout<<"Tracking result (output_stracks):"<<std::endl;
        for (const auto &track : tracks)
        {
            //打印跟踪信息
            std::cout<<"track_id: "<<track.track_id<<",Track  Bounding Box:["<<track.tlbr[0]<<","<<track.tlbr[1]<<","<<track.tlbr[2]<<","<<track.tlbr[3]<<"]"<<",sorce:"<<track.score<<std::endl;
            int x1=static_cast<int>(track.tlbr[0]);
            int y1=static_cast<int>(track.tlbr[1]);
            int x2=static_cast<int>(track.tlbr[2]);
            int y2=static_cast<int>(track.tlbr[3]);

            // 画出跟踪框
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

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;

    std::shared_ptr<cv_bridge::CvBridge> bridge_;
    std::mutex depth_mutex_;
    cv::Mat depth_image_;

    float fx_, fy_, cx_, cy_;

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
