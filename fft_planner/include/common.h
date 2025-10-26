#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <fftw3.h>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

namespace FFTPlanner {
typedef double DType;
typedef Eigen::Vector3d pose;
typedef Eigen::Vector3d velocity;
typedef Eigen::Vector3d acceleration;
typedef Eigen::Matrix<std::complex<DType>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix;
typedef Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> matrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> i_matrix;

struct ALLRes {
  Eigen::MatrixXd pl;
  Eigen::MatrixXd vl;
  Eigen::MatrixXd al;
  Eigen::VectorXd times;
  int deep_flag;
};

struct SVAResult {
  double s;               // 路程
  Eigen::VectorXd times;  // (ki+1)
  Eigen::MatrixXd p;      // (ki+1) x 3
  Eigen::MatrixXd v;      // (ki+1) x 3
  Eigen::MatrixXd a;      // (ki+1) x 3
  Eigen::VectorXd tao;    // (ki+1)
};

struct Header {
  uint32_t seq;
  DType timestamp;
  std::string frame_id;
};


struct Odom {
  Header header;
  std::string child_frame_id;
  Eigen::Vector3d pose;
  Eigen::Quaterniond orientation;
  std::array<DType, 36> pose_covariance;
  Eigen::Vector3d linear_twist;
  Eigen::Vector3d angular_twist;
  std::array<DType, 36> twist_covariance;
  bool is_valid{false};
};

struct Image  {
  Header header;
  uint32_t height;
  uint32_t width;
  std::string encoding;
  uint8_t is_bigendian;
  uint32_t step;
  std::vector<uint8_t> data;
  bool is_valid{false};
};
/**
 * @brief 将角标映射到矩阵边缘（按从矩阵中心发出的射线方向）
 *
 * @param matrixShape 矩阵的形状，(rows, cols)
 * @param index 输入角标 (x, y)，可能在矩阵外
 * @return Eigen::Vector2i 映射后的角标，位于矩阵边缘上
 */
Eigen::Vector2i Map2Edge(const Eigen::Vector2i& matrixShape,
                         const Eigen::Vector2i& index);

/**
 * @brief 计算最优中间速度 v1 的解析解
 *
 * @param p0 起始位置
 * @param v0 起始速度
 * @param a0 起始加速度
 * @param p1 中间点位置
 * @param p2 终止位置
 * @param v2 终止速度
 * @param a2 终止加速度
 * @param T1 第一段轨迹时间（从 p0 到 p1）
 * @param T2 第二段轨迹时间（从 p1 到 p2）
 * @return double 最优中间速度 v1
 */
double ComputeOptimalV1(double p0, double v0, double a0, double p1, double p2,
                        double v2, double a2, double T1, double T2);
/**
 * @brief 计算中间变量参数 a 和 b
 *
 * @param kalpha 系数 k_α
 * @param kbeta  系数 k_β
 * @param kgamma 系数 k_γ
 * @param balpha 偏置 b_α
 * @param bbeta  偏置 b_β
 * @param bgamma 偏置 b_γ
 * @param T      时间长度
 * @return std::pair<double, double> 返回 pair(a, b)
 */
std::pair<double, double> CalAB(double kalpha, double kbeta, double kgamma,
                                double balpha, double bbeta, double bgamma,
                                double T);
/**
 * @brief 将128*1024线的lidar深度数据转下采样到32*128
 *
 * @param data 原始深度图数据（以uint16_t为单位）
 * @param width 原始图像宽度（例如1024）
 * @param height 原始图像高度（例如128）
 * @param is_bigendian 字节序标志（true表示大端）
 * @return Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>
 * 下采样后的图像（32x128）
 */
matrix ConvertToDepthImage(const Image& msg);

complex_matrix my_ifft2(const complex_matrix &freq);

complex_matrix ifft2_exact(const complex_matrix &freq);

complex_matrix ifft2_stepwise(const complex_matrix &freq);

complex_matrix ifft2_direct(const complex_matrix& freq);
}  // namespace FFTPlanner