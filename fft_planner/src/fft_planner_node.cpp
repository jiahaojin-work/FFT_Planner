#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

#include "common.h"
#include "fft_planner.h"

FFTPlanner::Odom ConvertOdom(const nav_msgs::Odometry& msg) {
  FFTPlanner::Odom odom;
  odom.header.frame_id = msg.header.frame_id;
  odom.header.seq = msg.header.seq;
  odom.header.timestamp = msg.header.stamp.toSec();

  odom.child_frame_id = msg.child_frame_id;
  odom.pose =
      Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y,
                      msg.pose.pose.position.z);
  odom.orientation = Eigen::Quaterniond(msg.pose.pose.orientation.w, 
                                        msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z);
  std::copy(msg.pose.covariance.begin(), msg.pose.covariance.end(),
            odom.pose_covariance.begin());

  odom.linear_twist =
      Eigen::Vector3d(msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                      msg.twist.twist.linear.z);
  odom.angular_twist =
      Eigen::Vector3d(msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                      msg.twist.twist.angular.z);

  std::copy(msg.twist.covariance.begin(), msg.twist.covariance.end(),
            odom.twist_covariance.begin());
  odom.is_valid = true;
  return odom;
}

FFTPlanner::Image ConvertImage(const sensor_msgs::Image& msg) {
  FFTPlanner::Image image;
  image.header.frame_id = msg.header.frame_id;
  image.header.seq = msg.header.seq;
  image.header.timestamp = msg.header.stamp.toSec();

  image.height = msg.height;
  image.width = msg.width;
  image.encoding = msg.encoding;
  image.is_bigendian = msg.is_bigendian;
  image.step = msg.step;
  image.data = msg.data;
  image.is_valid = true;
  return image;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "fft_planner_node");
  ros::NodeHandle nh("~");

  std::string bag_path;

  // 必填参数：bag 文件路径
  FFTPlanner::Image fft_image;
  FFTPlanner::Odom fft_odom;
  if (!nh.getParam("bag", bag_path)) {
    ROS_FATAL("~bag param is required, e.g. _bag:=/path/to/file.bag");
    return 1;
  }

  // 打开 bag
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (const rosbag::BagException& e) {
    ROS_FATAL_STREAM("Failed to open bag: " << e.what());
    return 1;
  }

  rosbag::View view(bag);  // 读取全部 topic
  FFTPlanner::FFTPlanner planner(2);

  size_t count = 0;
  for (const rosbag::MessageInstance& m : view) {
    if (auto odom = m.instantiate<nav_msgs::Odometry>()) {
      ROS_INFO_STREAM("[Odom] topic=" << m.getTopic()
                                      << " t=" << odom->header.stamp.toSec()
                                      << " x=" << odom->pose.pose.position.x);
      fft_odom = ConvertOdom(*odom);
      ++count;
    } else if (auto img = m.instantiate<sensor_msgs::Image>()) {
      if (img->header.seq != 2300) {
        continue;
      }
      ROS_INFO_STREAM("[Image] topic="
                      << m.getTopic() << " t=" << img->header.stamp.toSec()
                      << " size=" << img->width << "x" << img->height);
      fft_image  = ConvertImage(*img);
      ++count;
    } else {
      ROS_DEBUG_STREAM("Skip topic=" << m.getTopic());  // 用 DEBUG 避免刷屏
    }
    if (fft_odom.is_valid && fft_image.is_valid) {
      auto depth_img = FFTPlanner::ConvertToDepthImage(fft_image);
      // std::cout << "test convert depth image" << depth_img.rows() << " " << depth_img.cols() << std::endl;
      // std::cout << depth_img << std::endl;
      planner.FFTParallel(fft_odom, depth_img, {200., 2., 1.5});

      break;
    }
  }
  ROS_INFO_STREAM("Processed " << count << " messages.");
  bag.close();
  return 0;
}
