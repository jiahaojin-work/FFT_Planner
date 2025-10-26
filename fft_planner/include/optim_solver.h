#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <set>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "common.h"
#include <ceres/ceres.h>

namespace FFTPlanner {

class LagrangeDynamicsSolver {
public:
  LagrangeDynamicsSolver(double r, const Eigen::Vector3d& a, const Eigen::Vector3d& b)
      : r_(r), a_(a), b_(b) {}

  // 主接口：给定初始角度，返回优化后的 [theta, phi]
  Eigen::Vector2d Solve(const Eigen::Vector3d& initial_guess) {
    double vars[2] = {initial_guess[0], initial_guess[1]};

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<LagrangeCostFunctor, 2, 2>(
            new LagrangeCostFunctor(r_, a_, b_)),
        nullptr, vars);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; // 设为true可查看优化过程
    // options.max_num_iterations = 100;
    // options.function_tolerance = 1e-6;
    // options.parameter_tolerance = 1e-6;

    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    options.max_num_iterations = 200;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type == ceres::CONVERGENCE) {
      std::cout << "[Ceres] Converged.\n";
    } else {
      std::cout << "[Ceres] Did not converge.\n";
    }

    return Eigen::Vector2d(vars[0], vars[1]);
  }

private:
  // 内部Functor
  struct LagrangeCostFunctor {
    LagrangeCostFunctor(double r, const Eigen::Vector3d& a, const Eigen::Vector3d& b)
        : r_(r), a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const vars, T* residuals) const {
      const T& theta = vars[0];
      const T& phi = vars[1];

      // 全局坐标
      T x = T(r_) * ceres::sin(theta) * ceres::cos(phi);
      T y = T(r_) * ceres::sin(theta) * ceres::sin(phi);
      T z = T(r_) * ceres::cos(theta);

      // 拉格朗日法方程
      T eq1 =
          (T(2) * T(a_[0]) * x + T(b_[0])) * (T(r_) * ceres::cos(theta) * ceres::cos(phi)) +
          (T(2) * T(a_[1]) * y + T(b_[1])) * (T(r_) * ceres::cos(theta) * ceres::sin(phi)) +
          (T(2) * T(a_[2]) * z + T(b_[2])) * (-T(r_) * ceres::sin(phi));

      T eq2 =
          (T(2) * T(a_[0]) * x + T(b_[0])) * (-T(r_) * ceres::sin(theta) * ceres::sin(phi)) +
          (T(2) * T(a_[1]) * y + T(b_[1])) * (T(r_) * ceres::sin(theta) * ceres::cos(phi));

      // residuals[0] = eq1;
      // residuals[1] = eq2;
      if (std::is_same<T, double>::value) {
      }
      residuals[0] = eq1;
      residuals[1] = eq2;
      return true;
    }

    double r_;
    Eigen::Vector3d a_, b_;
  };

  double r_;
  Eigen::Vector3d a_, b_;
};

class OptimSolver {
  public:
    OptimSolver() = default;
    ~OptimSolver() = default;
    OptimSolver(const Eigen::Vector3d &p0, const Eigen::Vector3d &a, const Eigen::Vector3d &b, double r,
                const matrix &safe, Eigen::Quaterniond rotation, double angle_ptx, double gmax, double gmin,
                const Eigen::Vector2i &shape, const Eigen::Vector2i &centre)
        : p0_(p0), a_(a), b_(b), r_(r), safe_(safe), rotation_(rotation), angle_ptx_(angle_ptx), gmax_(gmax),
          gmin_(gmin), shape_(shape), centre_(centre) {
        // r_g = shape[1] / 2 / 4 / 2
        r_g_ = static_cast<double>(shape[1]) / 2.0 / 4.0 / 2.0;

        // gmin_range = [r_g/2, shape[0] - r_g/2]
        gmin_range_.first = static_cast<int>(r_g_ / 2.0);
        gmin_range_.second = static_cast<int>(shape[0] - r_g_ / 2.0);

        // 初始化梯度矩阵为0
        gradient_ = Eigen::MatrixXd::Zero(shape[1], shape[1]);

        // 初始参考坐标设置为0
        refox_ = 0.0;
        refoy_ = 0.0;
    }

  public:
    void CalGmaxGmin() {
        gmax_ = gradient_.maxCoeff();
        gmin_ = gradient_.minCoeff();
    }

    std::pair<double, double> EquationDynamics(const Eigen::Vector2d &theta_phi);

    std::pair<int, int> SolveSafe();

    std::pair<int, int> Solve(const Eigen::Vector3d &initial_p);

  public:
    void set_gradient(const matrix &grad) {
        gradient_ = grad;
    }

    void cal_gmax_gmin() {
        if (gradient_.size() == 0) {
            gmax_ = 0.0;
            gmin_ = 0.0; // 空矩阵处理
            return;
        }

        // Eigen::Matrix 也可以直接用 .maxCoeff() 和 .minCoeff()
        gmax_ = gradient_.maxCoeff();
        gmin_ = gradient_.minCoeff();
    }

    void set_p0(const Eigen::Vector3d& p) {
      p0_ = p;
    }
    void set_a(const Eigen::Vector3d& a) {
      a_ = a;
    }
    void set_b(const Eigen::Vector3d& b) {
      b_ = b;
    }

    void set_safe(const matrix& safe) {
      safe_ = safe;
    }

    void set_rotation(const Eigen::Quaterniond rotation) {
      rotation_ = rotation;
    }

  private:
    double gmax_;
    double gmin_;
    double r_;
    Eigen::Vector3d a_;
    Eigen::Vector3d b_;
    Eigen::Vector3d p0_;

    matrix safe_;
    // double rotation_;
    Eigen::Quaterniond rotation_;
    double angle_ptx_;

    Eigen::Vector2i shape_;
    Eigen::Vector2i centre_;
    double r_g_;

    std::pair<int, int> gmin_range_;
    matrix gradient_;

    double refox_, refoy_;
};
} // namespace FFTPlanner