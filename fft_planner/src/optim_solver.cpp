#include "optim_solver.h"

#include "common.h"

namespace FFTPlanner {
std::pair<double, double> OptimSolver::EquationDynamics(
    const Eigen::Vector2d& theta_phi) {
  double theta = theta_phi[0];
  double phi = theta_phi[1];

  // 全局坐标转换
  double x = r_ * std::sin(theta) * std::cos(phi);
  double y = r_ * std::sin(theta) * std::sin(phi);
  double z = r_ * std::cos(theta);

  // 拉格朗日偏导数
  double eq1 =
      (2 * a_[0] * x + b_[0]) * (r_ * std::cos(theta) * std::cos(phi)) +
      (2 * a_[1] * y + b_[1]) * (r_ * std::cos(theta) * std::sin(phi)) +
      (2 * a_[2] * z + b_[2]) * (-r_ * std::sin(theta));

  double eq2 =
      (2 * a_[0] * x + b_[0]) * (-r_ * std::sin(theta) * std::sin(phi)) +
      (2 * a_[1] * y + b_[1]) * (r_ * std::sin(theta) * std::cos(phi));

  return std::make_pair(eq1, eq2);
}

std::pair<int, int> OptimSolver::SolveSafe() {
  int mx = refox_;
  int my = refoy_;

  if (mx < 0 || mx > shape_[0] - 1 || my < 0 || my > shape_[1] - 1) {
    Eigen::Vector2i result = Map2Edge(Eigen::Vector2i(shape_[0], shape_[1]),
                                      Eigen::Vector2i(mx, my));
    mx = result[0];
    my = result[1];
  }

  for (int tt = 0; tt < 100; ++tt) {
    if (safe_(mx, my)) {
      return {mx, my};
    }

    int ex_mx = mx;
    int ex_my = my;

    Eigen::Vector2d g(0, 0);
    if (0 < mx && mx < shape_[0] - 1 && 0 < my && my < shape_[1] - 1) {
      g[0] = (gradient_(mx + 1, my) - gradient_(mx - 1, my)) / 2.0;
      g[1] = (gradient_(mx, my + 1) - gradient_(mx, my - 1)) / 2.0;
    } else if (mx == 0 && 0 < my && my < shape_[1] - 1) {
      g[0] = (gradient_(mx + 1, my) - gradient_(mx, my));
      g[1] = (gradient_(mx, my + 1) - gradient_(mx, my - 1)) / 2.0;
    } else if (mx == shape_[0] - 1 && 0 < my && my < shape_[1] - 1) {
      g[0] = (gradient_(mx, my) - gradient_(mx - 1, my));
      g[1] = (gradient_(mx, my + 1) - gradient_(mx, my - 1)) / 2.0;
    } else if (0 < mx && mx < shape_[0] - 1 && my == 0) {
      g[0] = (gradient_(mx + 1, my) - gradient_(mx - 1, my)) / 2.0;
      g[1] = (gradient_(mx, my + 1) - gradient_(mx, my));
    } else if (0 < mx && mx < shape_[0] - 1 && my == shape_[1] - 1) {
      g[0] = (gradient_(mx + 1, my) - gradient_(mx - 1, my)) / 2.0;
      g[1] = (gradient_(mx, my) - gradient_(mx, my - 1));
    }

    double g_norm = g.norm();
    if (g_norm < 1e-6) continue;

    double d = r_g_ / (gmin_ - gmax_) * (gradient_(mx, my) - gmin_) + r_g_;
    mx += std::round(g[0] / g_norm * d);
    my += std::round(g[1] / g_norm * d);

    if (std::abs(mx - ex_mx) < 1 && std::abs(my - ex_my) < 1) {
      return {mx, my};
    }

    if (mx < 0 || mx > shape_[0] - 1 || my < 0 || my > shape_[1] - 1) {
      Eigen::Vector2i result = Map2Edge(Eigen::Vector2i(shape_[0], shape_[1]),
                                        Eigen::Vector2i(mx, my));
      mx = result[0];
      my = result[1];
    }
  }

  return {mx, my};
}


std::pair<int, int> OptimSolver::Solve(const Eigen::Vector3d& initial_p) {
  // TODO: fsolve 用什么替换
  LagrangeDynamicsSolver solver(r_, a_, b_);
  auto res = solver.Solve(initial_p);
  // auto res = solver.solve({initial_p(0), initial_p(1)});
  std::cout << "test ceres res " << res[0] << " " << res[1] << std::endl;
  DType theta_sol = res[0];
  DType phi_sol = res[1];

  Eigen::Vector3d po(
        r_ * std::sin(theta_sol) * std::cos(phi_sol),
        r_ * std::sin(theta_sol) * std::sin(phi_sol),
        r_ * std::cos(theta_sol));

  Eigen::Vector3d po_local = rotation_.inverse() * po;

  Eigen::Vector3d eo_local = po_local.normalized();

  DType angle_horizontal1 = std::atan2(eo_local.y(), eo_local.x()); 
  DType refoy_f = (centre_.y() - angle_horizontal1 / angle_ptx_);     

  DType angle_vertical1 = std::acos(eo_local.z());                  
  DType refox_f = (centre_.x() + (angle_vertical1 - M_PI / 2.0) / angle_ptx_);

  refox_ = static_cast<int>(std::lround(refox_f));
  refoy_ = static_cast<int>(std::lround(refoy_f));

  auto [mx, my] = SolveSafe();
  std::cout << "mx " << mx << " my " << my << std::endl;
  return {mx, my};
}

}  // namespace FFTPlanner