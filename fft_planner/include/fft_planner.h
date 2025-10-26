#pragma once

#include <complex>
#include <fftw3.h>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unsupported/Eigen/FFT>

#include "common.h"
#include "optim_solver.h"

#include <nlohmann/json.hpp>
#include <cmath>

namespace FFTPlanner {


// struct PVA {
//   Eigen::Vector3d p;
//   Eigen::Vector3d v;
//   Eigen::Vector3d a;
// };

struct PolyParams {
  Eigen::Vector3d alpha;
  Eigen::Vector3d beta;
  Eigen::Vector3d gamma;
};

// struct TrajResult {
//   std::vector<Eigen::Vector3d> pl;
//   std::vector<Eigen::Vector3d> vl;
//   std::vector<Eigen::Vector3d> al;
//   std::vector<double> times;
// };

// struct Params {
//   Eigen::Vector3d alpha, beta, gamma;
// };

struct BestIdxEndResult {
  Eigen::MatrixXd p_sol;
  Eigen::MatrixXd pl;
  Eigen::MatrixXd vl;
  Eigen::MatrixXd al;
  Eigen::MatrixXd times;
  double T1{0.0};
  double T2{0.0};
  Eigen::Vector3d v1;
  Eigen::VectorXd tao;
  double s_all{0.0};
};

class FFTPlanner {
  public:
    FFTPlanner() = delete;
    FFTPlanner(int mode = 0);
    ~FFTPlanner() = default;

  public:
    complex_matrix LoadCSV(const std::string &file);

    void SaveComplexMatrixToJson(const complex_matrix &mat, const std::string &filename) {
        using json = nlohmann::json;
        json j;
        int rows = mat.rows();
        int cols = mat.cols();
        j["rows"] = rows;
        j["cols"] = cols;

        // 存储为 [[ [real, imag], [real, imag], ... ], ... ]
        for (int i = 0; i < rows; i++) {
            json row = json::array();
            for (int jcol = 0; jcol < cols; jcol++) {
                row.push_back({mat(i, jcol).real(), mat(i, jcol).imag()});
            }
            j["data"].push_back(row);
        }
        std::cout << "json file name " << std::endl;
        std::ofstream file(filename);
        file << j.dump(2); // pretty print
        file.close();
    }

    std::complex<double> ParseComplex(const std::string &s) {
        double re, im;
        char ch1, ch2, j;
        std::stringstream ss(s);
        ss >> ch1 >> re >> ch2 >> im >> j; // e.g. "(16.9+240.4j)"

        if (ch2 == '-') {
            im = -im;
        }
        return {re, im};
    }

    cv::Mat EigenToCv(const complex_matrix &eigMat) {
        int rows = eigMat.rows();
        int cols = eigMat.cols();

        // OpenCV 复数矩阵 2 通道 double
        cv::Mat cvMat(rows, cols, CV_64FC2);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const std::complex<double> &val = eigMat(i, j);
                cvMat.at<cv::Vec2d>(i, j)[0] = val.real();
                cvMat.at<cv::Vec2d>(i, j)[1] = val.imag();
            }
        }
        return cvMat;
    }

    matrix Downsample(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, int dtsep);

    std::vector<std::pair<int, int>> LocalMaximaAxis0(const Eigen::MatrixXd &mat, int order = 1);
    std::vector<std::pair<int, int>> LocalMaximaAxis1(const Eigen::MatrixXd &mat, int order = 1);

    void FFTProcess(std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> &processed,
                    const Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> &image, const int j);

    ALLRes FFTParallel(const Odom &odom, const Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> &image,
                const Eigen::Vector3d &target);

    Eigen::MatrixXi ArgrelextremaGreater(const Eigen::MatrixXd &mat, int axis, int order);
    Eigen::MatrixXi Argrelextrema(const Eigen::MatrixXd &A, bool find_max = true, int axis = 0, int order = 1);
    Eigen::MatrixXi GetIntersectionMatrix(const std::set<std::pair<int, int>> &set_max0,
                                          const std::set<std::pair<int, int>> &set_max1);

        complex_matrix fft2(const matrix &gray_image, int Hshape);

    complex_matrix ifft2(const complex_matrix &freq);
    // void Reset();

    // double sphericalDistance(double theta1, double phi1, double theta2,
    //                          double phi2) const;

    std::pair<double, double> CaLAB(double kalpha, double kbeta, double kgamma, double balpha, double bbeta,
                                    double bgamma, double T);
    SVAResult CalSVA(const Eigen::Vector3d &alpha, const Eigen::Vector3d &beta, const Eigen::Vector3d &gamma,
                      const Eigen::Vector3d &a0, const Eigen::Vector3d &v0, const Eigen::Vector3d &p0, double T) const;

    SVAResult CalSVAT(const Eigen::Vector3d &alpha, const Eigen::Vector3d &beta, const Eigen::Vector3d &gamma,
                const Eigen::Vector3d &a0, const Eigen::Vector3d &v0, const Eigen::Vector3d &p0, double t);

    Eigen::Vector3d ComputeOptimalV1(const Eigen::Vector3d &p0, const Eigen::Vector3d &v0, const Eigen::Vector3d &a0,
                                       const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3d &v2,
                                       const Eigen::Vector3d &a2, double T1, double T2);

    PolyParams CalFullyDefinedParam(const Eigen::Vector3d& p0,
                                       const Eigen::Vector3d& v0,
                                       const Eigen::Vector3d& a0,
                                       const Eigen::Vector3d& pf,
                                       const Eigen::Vector3d& vf1,
                                       const Eigen::Vector3d& af, double T);

    double CalFullyDefinedJ(const Eigen::Vector3d& alpha,
                               const Eigen::Vector3d& beta,
                               const Eigen::Vector3d& gamma, double T);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>
    CalForwardKinematicsDivide(Eigen::MatrixXd pl, Eigen::MatrixXd vl, Eigen::MatrixXd al, Eigen::Vector3d p1,
                                  Eigen::VectorXd times, Eigen::VectorXd tao, double T1, double T2,
                                  bool p1_safe = true);

    Eigen::VectorXd ConcatVector(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, int skip = 0,
                                 double offset = 0.0) {
        int size1 = v1.size();
        int size2 = v2.size() - skip;
        Eigen::VectorXd v(size1 + size2);

        v.head(size1) = v1;
        v.tail(size2) = v2.tail(size2).array() + offset; // 偏移量可选
        return v;
    }

    Eigen::MatrixXd ConcatMatrix(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2, int skip = 0) {
        int rows1 = m1.rows();
        int rows2 = m2.rows() - skip;
        int cols = m1.cols();

        Eigen::MatrixXd m(rows1 + rows2, cols);
        m.topRows(rows1) = m1;
        m.bottomRows(rows2) = m2.bottomRows(rows2);
        return m;
    }
    // def find_nearest_margin(self, matrix, ref_index, terminal_index, r_safe, r_extreme):

    Eigen::Vector2i FindNearestMargin(matrix &matrix, const Eigen::VectorXi &ref_idx, const Eigen::Vector2i &terminal_index,
                            DType r_safe, DType r_extreme);

    BestIdxEndResult CalBestIdxEnd(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& v0,
                                      const Eigen::Vector3d& a0,
                                      const Eigen::Vector3d& p2_in,
                                      Eigen::Vector3d v2_in);

    Eigen::Vector2i FindNearestOneMargin(const matrix& matrix, const Eigen::VectorXi ref_index);
    // Eigen::Vector2i findNearestOne(
    //     const Eigen::Ref<const Eigen::MatrixXi>& matrix,
    //     const Eigen::Vector2i& ref_index,
    //     const Eigen::Vector2i& terminal_index) const;

    // Eigen::Vector2i findNearestMargin(
    //     const Eigen::Ref<const Eigen::MatrixXi>& matrix,
    //     const Eigen::Vector2i& ref_index, const Eigen::Vector2i&
    //     terminal_index, double r_safe, double r_extreme) const;
    // Eigen::Vector2i findNearestOneOrigin(
    //     const Eigen::Ref<const Eigen::MatrixXi>& matrix,
    //     Eigen::Vector2i ref_index) const;

  private:
    std::vector<std::thread> jobs_;
    std::mutex mutex_;
    // 参数配置
    int mode_;
    double max_spd_ = 20.0;
    double max_acc_ = 15.0;
    double exp_spd_ = 4.0;
    double kt_ = 0.25;

    double epsilon_ = 3.71;
    int ang_acc_ = 1;
    int dtsep_ = 2;
    double replan_time_ = 0.5;

    Eigen::Vector2d fov_;
    Eigen::Vector2i shape_;
    Eigen::Vector2i mx_centre_;

    int iter_;
    double dis_max_;
    std::vector<double> cutoff_;
    Eigen::Vector2i r_;
    double angle_ptx_;
    int path_len_;

    Eigen::MatrixXd w_costs_;

    std::vector<complex_matrix> H_S_;
    std::vector<complex_matrix> H_D_;

    int Hshape_;
    // std::vector<Eigen::Tensor<bool, 3> safe_;  // shape: path_len × shape[0] ×
    // shape[1]
    std::vector<matrix> safe_;
    Eigen::Quaterniond rotation_;
    Eigen::Vector3d p0_;
    Eigen::Vector3d linar_v_;
    Eigen::Vector3d linar_a_;

    Eigen::Vector3d target_direction_;
    double target_distance_;
    Eigen::Vector3d target_;
    Eigen::Vector3d target_vector_;
    bool approaching_;

    Eigen::Vector3d last_end_point_;
    Eigen::Vector3d last_end_val_;
    Eigen::Vector3d last_position_;

    OptimSolver solver_;

    double last_time_;
    double since_end_time_;
    bool last_invalid_ = false;
    Eigen::Vector3d last_linear_velocity_;
};

} // namespace FFTPlanner