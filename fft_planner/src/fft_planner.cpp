#include "fft_planner.h"

namespace FFTPlanner {
constexpr DType kFTotalThres = 0.5;

FFTPlanner::FFTPlanner(int mode)
    : mode_(mode), max_spd_(20.0), max_acc_(15.0), exp_spd_(4.0), kt_(0.25), epsilon_(3.71), ang_acc_(1), dtsep_(2),
      replan_time_(0.5), fov_(45. * M_PI / 180., 180. * M_PI / 180.), shape_(32, 128), mx_centre_(16, 64), iter_(5),
      dis_max_(10.0), cutoff_({5.65, 10.0}), r_(3, 2), // 7/2=3, 4/2=2
      angle_ptx_(1.40625 * M_PI / 180.0), path_len_(2), rotation_(Eigen::Quaterniond::Identity()),
      p0_(Eigen::Vector3d::Zero()), linar_v_(Eigen::Vector3d::Zero()), linar_a_(Eigen::Vector3d::Zero()),
      target_direction_(1.0, 0.0, 0.0), target_distance_(0.0), target_(10.0, 0.0, 0.0), target_vector_(10.0, 0.0, 0.0),
      approaching_(false), last_end_point_(Eigen::Vector3d::Zero()), last_end_val_(Eigen::Vector3d::Zero()),
      last_position_(Eigen::Vector3d::Zero()), last_time_(0.0), since_end_time_(0.0),
      last_linear_velocity_(Eigen::Vector3d::Zero()) {
    // 初始化 w_costs
    w_costs_ = Eigen::MatrixXd::Zero(path_len_, 4);
    for (int i = 0; i < path_len_; ++i) {
        w_costs_(i, 0) = 1.0;
        w_costs_(i, 1) = 1.0;
        w_costs_(i, 2) = 1.0;
        w_costs_(i, 3) = 0.5;
        w_costs_.row(i) /= w_costs_.row(i).sum();
    }
    H_S_.clear();
    H_D_.clear();
    // 加载 H_S 滤波器
    for (int i = 0; i < 2; ++i) {
        std::string path = "/home/jack/catkin_ws/src/fft_planner/py/H" + std::to_string(i + 1) + "-f-S.csv";
        try {
            H_S_.push_back(LoadCSV(path));
        } catch (...) {
            std::cerr << "Error loading file: " << path << std::endl;
        }
    }

    // 加载 H_D 滤波器
    for (int i = 0; i < 2; ++i) {
        std::string path = "/home/jack/catkin_ws/src/fft_planner/py/H" + std::to_string(i + 1) + "-f-D.csv";
        try {
            H_D_.push_back(LoadCSV(path));
        } catch (...) {
            std::cerr << "Error loading file: " << path << std::endl;
        }
    }
    for (const auto &hs : H_S_) {
        std::cout << "---------------- HS ------------" << std::endl;
        std::cout << hs << std::endl;
    }
    for (const auto &hd : H_D_) {
        std::cout << "---------------- HD ------------" << std::endl;
        std::cout << hd << std::endl;
    }
    Hshape_ = H_D_[0].rows(); // 假设为方阵

    // 初始化 safe
    // TODO: safe改造
    // for (int i = 0; i < path_len_; ++i) {
    //     safe_ = Eigen::MatrixXd::Zero(shape_[0], shape_[1]);
    // }
    safe_.resize(path_len_);
    // 初始化 solver
    // TODO: 初始化函数适配
    solver_ = OptimSolver(p0_, Eigen::Vector3d::Zero(),
                              Eigen::Vector3d::Zero(), cutoff_[0], safe_[0],
                              rotation_, angle_ptx_, 0.0, 0.0, shape_,
                              mx_centre_);

    std::cout << "初始化完成" << std::endl;
}

complex_matrix FFTPlanner::LoadCSV(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;
    std::vector<std::complex<double>> values;
    size_t rows = 0;
    size_t expected_cols = 0;

    while (std::getline(in, line)) {
        if (line.empty())
            continue; // 跳过空行

        std::stringstream ss(line);
        std::string cell;
        size_t cols = 0;

        while (std::getline(ss, cell, ',')) {
            if (cell.empty())
                continue;
            try {
                values.push_back(ParseComplex(cell));
            } catch (const std::exception &e) {
                std::cerr << "Error parsing cell '" << cell << "' at row " << rows << ": " << e.what() << std::endl;
                throw;
            }
            ++cols;
        }

        if (rows == 0)
            expected_cols = cols;
        else if (cols != expected_cols) {
            throw std::runtime_error("Inconsistent columns in CSV at row " + std::to_string(rows));
        }
        ++rows;
    }

    if (rows == 0 || expected_cols == 0) {
        throw std::runtime_error("Empty CSV file: " + path);
    }
    return Eigen::Map<const complex_matrix>(values.data(), rows, expected_cols);
}

ALLRes FFTPlanner::FFTParallel(const Odom &odom, const Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> &image,
                        const Eigen::Vector3d &target) {
    ALLRes all_res;
    BestIdxEndResult bi_res;
    DType err_ptx;
    int deep_flag = 0;
    DType imgidx = odom.header.timestamp;
    rotation_ = odom.orientation;
    Eigen::Vector3d euler = odom.orientation.toRotationMatrix().eulerAngles(0, 1, 2);
    std::cout << "test euler " << euler << std::endl;
    DType f_pitch = std::max(0.0, 1 - std::abs(euler[1]) / fov_[0]);
    std::cout << "f pitch test " << fov_[0] << " " << f_pitch << std::endl;
    DType a_roll =
        fov_[0] * fov_[1] - ((1.0 - std::cos(euler[0])) / (std::sin(euler[0]) * std::cos(euler[0]))) *
                                (fov_[0] * fov_[0] + fov_[1] * fov_[1] - 2.0 * fov_[0] * fov_[0] * std::cos(euler[0]));
    DType f_roll = std::max(0.0, a_roll / (fov_[0] * fov_[1]));
    DType f_total = f_pitch * f_roll;
    Eigen::MatrixXi ref_index = Eigen::MatrixXi::Ones(path_len_, 2);

    std::cout << "f total test " << a_roll << " " << f_roll << " " << f_total << std::endl;
    if (f_total < kFTotalThres) {
        last_time_ = -1.0;
        last_linear_velocity_ = Eigen::Vector3d::Zero();
        last_end_point_ = Eigen::Vector3d::Zero();
        last_end_val_ = Eigen::Vector3d::Zero();
        last_position_ = Eigen::Vector3d::Zero();
        last_invalid_ = false;
        std::cout << "f float < kFTotalThres" << std::endl;
        // TODO: build a return value
        return all_res;
    }
    p0_ = odom.pose;
    linar_v_ = odom.linear_twist;
    if (last_invalid_) {
        DType dt = odom.header.timestamp - last_time_;
        linar_a_ = (linar_v_ - last_linear_velocity_) / dt;
    } else {
        linar_a_ = Eigen::Vector3d::Zero();
    }
    target_ = target;
    target_vector_ = target - p0_;
    target_distance_ = target_vector_.norm();
    if (target_distance_ > 0.7) {
        target_direction_ = target_vector_ / target_distance_;
    } else {
        last_invalid_ = false;
        last_time_ = -1.0;
        last_linear_velocity_ = Eigen::Vector3d::Zero();
        last_end_point_ = Eigen::Vector3d::Zero();
        last_end_val_ = Eigen::Vector3d::Zero();
        last_position_ = Eigen::Vector3d::Zero();
        std::cout << "target distance " << target_distance_ << " < 0.7  " << std::endl;
        return all_res;
        // TODO: add a return value
    }
    target_direction_ = rotation_.inverse() * target_direction_;
    if (since_end_time_ > 0.0 && std::abs(since_end_time_ - imgidx) > replan_time_) {
        since_end_time_ = -1.0;
        last_end_point_ << 0.0, 0.0, 0.0;
    }
    Eigen::Vector2i ref_endpt;
    if (last_invalid_) {
        if (since_end_time_ < 0.0) {
            since_end_time_ = imgidx;
        }
        auto end_point1 = last_end_point_ + last_end_val_ * (imgidx - last_time_) - (p0_ - last_position_);
        auto po_local = rotation_.inverse() * end_point1;
        auto eo_local = po_local.normalized();
        DType angle_horizontal1 = std::atan(eo_local[1] / eo_local[0]);
        DType refoy = mx_centre_[1] - angle_horizontal1 / angle_ptx_;
        DType angle_vertical1 = std::acos(eo_local[2]);
        DType refox = mx_centre_[0] + angle_vertical1 / angle_ptx_;

        ref_endpt << static_cast<int>(std::round(refox)), static_cast<int>(std::round(refoy));
        ref_endpt = Map2Edge(shape_, ref_endpt);
    } else {
        ref_endpt << 0, 0;
    }

    // 1. z 轴俯仰角
    DType pitch_angle = std::acos(target_direction_[2]) - M_PI / 2.0;

    // 2. xy 平面水平角
    DType horizontal_angle = std::atan(target_direction_[1] / target_direction_[0]);
    if (target_direction_[0] < 0) {
        horizontal_angle = -horizontal_angle; // 对应 np.sign
    }
    // 3. 转整数坐标
    Eigen::Vector2i ref;
    ref << static_cast<int>(std::round(mx_centre_[0] + pitch_angle / angle_ptx_)),
        static_cast<int>(std::round(mx_centre_[1] - horizontal_angle / angle_ptx_));
    ref = Map2Edge(shape_, ref);
    std::cout << "test ref " << ref[0] << " " << ref[1] << std::endl;

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> processed(path_len_);
    jobs_.reserve(path_len_);
    std::cout << "test mode " << mode_ << std::endl;
    if (mode_ == 0 || mode_ == 1) {
        jobs_.reserve(path_len_);
        for (int j = 0; j < path_len_; ++j) {
            jobs_.emplace_back(&FFTPlanner::FFTProcess, this, std::ref(processed), std::cref(image), j);
        }
        for (auto &job : jobs_) {
            if (job.joinable()) {
                job.join();
            }
        }
        for (const auto &metric : processed) {
            std::cout << metric << std::endl;
        }
    } else if (mode_ == 2 || mode_ == 3) {
        for (int j = 0; j < path_len_; ++j) {
            FFTProcess(processed, image, j);
            for (const auto &metric : processed) {
                std::cout << metric << std::endl;
            }
        }
    }
    if (target_distance_ < cutoff_[1] && safe_[1](ref[0], ref[1]) == 1) {
    // if (true) {
        DType c = target_distance_;
        DType theta = std::acos(linar_v_.dot(target_vector_) / (linar_v_.norm() * target_vector_.norm()));
        DType s = c * theta / (2.0 * std::sin(theta / 2.0) + 1e-6);
        if (!(0 < s && s < c * M_PI)) {
            s = c * 1.1;
        }
        DType T = std::sqrt(2.0 * s / max_acc_);
        auto param = CalFullyDefinedParam(Eigen::Vector3d::Zero(), linar_v_, linar_a_, target_vector_,
                                          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), T);
        std::cout << "alpha " << param.alpha << std::endl;
        std::cout << "beta " << param.beta << std::endl;
        std::cout << "gamma " << param.gamma << std::endl;

        
        auto sva = CalSVA(param.alpha, param.beta, param.gamma, linar_a_, linar_v_, Eigen::VectorXd::Zero(3), T);
        if (sva.tao.maxCoeff() > 1 + 1e-2) {
            auto res = CalForwardKinematicsDivide(sva.p, sva.v, sva.a, sva.p.row(sva.p.rows() - 1), sva.times, sva.tao,
                                                  sva.times[sva.times.size() - 1], -1, true);
            // return res;
            // TODO: return val
            all_res.pl = std::get<0>(res);
            all_res.vl = std::get<1>(res);
            all_res.al = std::get<2>(res);
            all_res.times = std::get<3>(res);
            all_res.deep_flag = 1;
            return all_res;
        }
    }

    if (safe_[0].maxCoeff() < 0.01) {
        deep_flag = 0;
    } else {
        deep_flag = 2;
        DType cost2, cost3, cost4;
        Eigen::MatrixXi ref_index_theory = Eigen::MatrixXi::Ones(path_len_, 2);
        Eigen::MatrixXd pf_all_wf = Eigen::MatrixXd::Zero(2, 3);
        Eigen::Vector3d pf2;
        Eigen::Vector3d v2;
        if (!approaching_) {
            if ((!(ref_endpt.array() == 0).all()) && safe_[1](ref_endpt[0], ref_endpt[1]) &&
                safe_[0](ref_endpt[0], ref_endpt[1])) {
                ref_index_theory.row(1) = ref_endpt;
            } else if (safe_[1](ref[0], ref[1]) && safe_[0](ref[0], ref[1])) {
                ref_index_theory.row(1) = ref;
            } else {
                // TODO: define all possible lines
                // define all costs
                Eigen::MatrixXd all_costs;
                Eigen::MatrixXd all_possible_lines;
                if (processed[1].size() > 0) {
                    DType c_max = processed[1].colwise().maxCoeff()(2);
                    if (c_max > 1.0) {
                        c_max = 1.0;
                    }
                    if (!(ref_endpt.array() == 0).all()) {
                        w_costs_(1, 3) = 0.5 * (1 - abs(since_end_time_ - imgidx) / replan_time_);
                        w_costs_.row(1) = w_costs_.row(1) / w_costs_.row(1).sum();
                    }

                    for (int i = 0; i < processed[1].rows(); ++i) {
                        DType mx = processed[1](i, 0);
                        DType my = processed[1](i, 1);
                        DType c = processed[1](i, 2);

                        // (1) 动力学成本
                        DType angle_horizontal = angle_ptx_ * (mx_centre_[1] - my);
                        DType angle_vertical = 3.1415926 / 2 - angle_ptx_ * (mx_centre_[0] - mx);
                        DType xx = cutoff_[1] * std::sin(angle_vertical) * std::cos(angle_horizontal);
                        DType yy = cutoff_[1] * std::sin(angle_vertical) * std::sin(angle_horizontal);
                        DType zz = cutoff_[1] * std::cos(angle_vertical);

                        Eigen::Vector3d local_point(xx, yy, zz);  // 局部坐标
                        Eigen::Vector3d p2_hat = rotation_ * local_point;
                        auto v2_hat = p2_hat;
                        DType theta_hat =
                            std::acos(linar_v_.dot(v2_hat) / (linar_v_.norm() * v2_hat.norm()));
                        DType s_hat = cutoff_[1] * theta_hat / (2.0 * sin(theta_hat / 2.0));
                        DType T_hat = 2.0 * s_hat / (exp_spd_ + linar_v_.norm());
                        auto param = CalFullyDefinedParam(Eigen::Vector3d::Zero(), linar_v_, linar_a_, p2_hat, v2_hat,
                                                          Eigen::Vector3d::Zero(), T_hat);
                        DType J_hat = CalFullyDefinedJ(param.alpha, param.beta, param.gamma, T_hat);

                        // (2) 空旷成本
                        if (c >= 1.0) {
                            cost2 = 1e-6;
                        } else {
                            cost2 = c_max - c;
                        }

                        // (3) 参考方向成本
                        cost3 = (Eigen::Vector2d(ref[0] - mx, ref[1] - my)).norm();

                        // (4) 上时刻规划方向成本
                        if (!(ref_endpt.array() == 0).all()) {
                            cost4 = (Eigen::Vector2d(ref_endpt[0] - mx, ref_endpt[1] - my)).norm();
                        } else {
                            cost4 = 1;
                        }
                        Eigen::RowVectorXd new_rowx(4);
                        new_rowx << J_hat, cost2, cost3, cost4;
                        all_costs.conservativeResize(all_costs.rows() + 1, 4);
                        all_costs.row(all_costs.rows() - 1) = new_rowx;

                        Eigen::RowVector2d new_row(mx, my);
                        all_possible_lines.conservativeResize(all_possible_lines.rows() + 1, 2);
                        all_possible_lines.row(all_possible_lines.rows() - 1) = new_row;
                    }

                }
                if (all_possible_lines.rows() == 0) {
                    ref_index_theory.row(1) << -1, -1;
                }  else if (all_possible_lines.rows() > 1) {

                    Eigen::MatrixXd normalized = all_costs.array().rowwise() / all_costs.colwise().maxCoeff().array();
                    auto quantile_sums = normalized * w_costs_.row(1).transpose();
                    Eigen::Index best_idx;
                    quantile_sums.minCoeff(&best_idx);
                    ref_index_theory.row(1) = all_possible_lines.row(best_idx).cast<int>();
                } else {
                    ref_index_theory.row(1) = all_possible_lines.row(0).cast<int>();
                }
            }
            std::cout << "ref index theory " << ref_index_theory << std::endl;
            if ((ref_index_theory.row(1).array() == -1).any()) {
                deep_flag = 1;
            } else {
                ref_index.row(1) = FindNearestMargin(safe_[1], ref_index_theory.row(1), ref, r_[1], shape_[1] / 18.0);
                std::cout << "ref index " << ref_index << std::endl;
            }

            if (!(ref_index.row(1).array() == -1).any()) {
                DType angle_horizontal = angle_ptx_ * (mx_centre_[1] - ref_index(1, 1));
                DType angle_vertical = 3.1415926535897932 / 2.0 - angle_ptx_ * (static_cast<DType>(mx_centre_[0]) -
                                                                    static_cast<DType>(ref_index(1, 0)));

                DType xx = cutoff_[1] * std::sin(angle_vertical) * std::cos(angle_horizontal);
                DType yy = cutoff_[1] * std::sin(angle_vertical) * std::sin(angle_horizontal);
                DType zz = cutoff_[1] * std::cos(angle_vertical);
                Eigen::Vector3d pf2_local2;
                pf2_local2 << xx, yy, zz;
                pf2 = rotation_ * pf2_local2;
                pf_all_wf.row(1) = pf2;

                auto future_target = target_vector_ - pf2;
                DType future_dis = future_target.norm();

                if (target_distance_ > cutoff_[cutoff_.size() - 1]) {
                    double scale = std::min(exp_spd_, std::sqrt(2.0 * max_acc_ * future_dis));
                    v2 = pf2.normalized() * scale; // normalized() 返回单位向量
                } else {
                    v2 = Eigen::Vector3d::Zero(); // 零向量
                }
            } else {
                pf2 << -1, -1, -1;
                v2 = Eigen::Vector3d::Zero();
            }
        } else {
            pf2 = target_vector_;
            pf_all_wf.row(1) = pf2;
            ref_index.row(1) = ref;
            v2 = Eigen::Vector3d::Zero();
        }
        bi_res = CalBestIdxEnd(Eigen::Vector3d::Zero(), linar_v_, linar_a_, pf2, v2);
        std::cout << "CalBestIdxEnd res p sol " << bi_res.p_sol << std::endl;
        std::cout << "CalBestIdxEnd res pl " << bi_res.pl << std::endl;
        std::cout << "CalBestIdxEnd res vl " << bi_res.vl << std::endl;
        std::cout << "CalBestIdxEnd res al " << bi_res.al << std::endl;
        std::cout << "CalBestIdxEnd res times " << bi_res.times << std::endl;
        std::cout << "CalBestIdxEnd res T1 " << bi_res.T1 << std::endl;
        std::cout << "CalBestIdxEnd res T2 " << bi_res.T2 << std::endl;
        std::cout << "CalBestIdxEnd res v1 " << bi_res.v1 << std::endl;
        std::cout << "CalBestIdxEnd res tao " << bi_res.tao << std::endl;
        std::cout << "CalBestIdxEnd res s all " << bi_res.s_all << std::endl;

        pf_all_wf.row(0) = bi_res.p_sol.col(0).transpose();
        auto p_ref_local = rotation_.inverse() * bi_res.p_sol.col(0);
        auto e_ref_local = p_ref_local.normalized();
        DType angle_horizontal1 = std::atan(e_ref_local(1) / e_ref_local(0));
        ref_index_theory(0, 1) = std::lround(mx_centre_(1) - angle_horizontal1 / angle_ptx_);
        DType angle_vertical1 = std::acos(e_ref_local(2));
        ref_index_theory(0, 0) = mx_centre_(0) + (angle_vertical1 - M_PI / 2.0) / angle_ptx_;
        ref_index.row(0) = FindNearestOneMargin(safe_[0], ref_index_theory.row(0));
        std::cout << "ref index " << ref_index << std::endl;
        err_ptx = (ref_index.row(0) - ref_index_theory.row(0)).norm();

        if ((ref_index.row(0).array() == -1).any()) {
            deep_flag = 0;
        }
    }
    std::cout << "deep flag test " << deep_flag << std::endl;
    if (deep_flag == 0) {
        last_time_ = -1.0;
        last_linear_velocity_ = Eigen::Vector3d::Zero();
        last_end_point_ = Eigen::Vector3d::Zero();
        last_end_val_ = Eigen::Vector3d::Zero();
        last_position_ = Eigen::Vector3d::Zero();
        return all_res;
    }
    if (bi_res.s_all > cutoff_[deep_flag - 1] * M_PI / 2.0) {
        last_time_ = -1.0;
        last_linear_velocity_ = Eigen::Vector3d::Zero();
        last_end_point_ = Eigen::Vector3d::Zero();
        last_end_val_ = Eigen::Vector3d::Zero();
        last_position_ = Eigen::Vector3d::Zero();
        return all_res;
    }
    Eigen::VectorXd p1_wf;
    if (err_ptx) {
        DType angle_horizontal = angle_ptx_ * (mx_centre_(1) - ref_index(0, 1));
        DType angle_vertical = M_PI / 2.0 - angle_ptx_ * (mx_centre_(0) - ref_index(0, 0));

        double xx = cutoff_[0] * std::sin(angle_vertical) * std::cos(angle_horizontal);
        double yy = cutoff_[0] * std::sin(angle_vertical) * std::sin(angle_horizontal);
        double zz = cutoff_[0] * std::cos(angle_vertical);

        p1_wf = rotation_ * Eigen::Vector3d(xx, yy, zz);
    }
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> kinematics_res;
    if (err_ptx || bi_res.tao.maxCoeff() > 1.0) {
        if (deep_flag == path_len_) {
            auto kinematics_res = CalForwardKinematicsDivide(bi_res.pl, bi_res.vl, bi_res.al, p1_wf, bi_res.times,
                                                             bi_res.tao, bi_res.T1, bi_res.T2, err_ptx == 0);
        } else {
            auto kinematics_res = CalForwardKinematicsDivide(bi_res.pl, bi_res.vl, bi_res.al, p1_wf, bi_res.times,
                                                             bi_res.tao, bi_res.T1, -1, err_ptx == 0);
        }
    }
    last_time_ = imgidx;
    last_linear_velocity_ = linar_v_;
    last_end_point_ = std::get<0>(kinematics_res).row(std::get<0>(kinematics_res).rows() - 1);
    last_position_ = p0_;
    all_res.pl = std::get<0>(kinematics_res);
    all_res.vl = std::get<1>(kinematics_res);
    all_res.al = std::get<2>(kinematics_res);
    all_res.times = std::get<3>(kinematics_res);
    all_res.deep_flag = deep_flag;
    return all_res;
}

Eigen::Vector2i FFTPlanner::FindNearestOneMargin(const matrix& matrix, const Eigen::VectorXi ref_index) {
    // 检查输入合法性
    std::cout << "matrix test " << matrix << std::endl;
    std::cout << "ref index test " << ref_index << std::endl;
    if (ref_index[0] == -1 || ref_index[1] == -1)
        return Eigen::Vector2i(-1, -1);

    auto ref_idx = Map2Edge(shape_, Eigen::Vector2i(ref_index(0), ref_index(1)));
    if (matrix(ref_idx(0), ref_idx(1)) == 1) {
        std::cout << "tes t1" << std::endl;
        return ref_idx;
    }

    std::set<std::pair<int, int>> visited;
    std::queue<std::pair<int, int>> queue;

    queue.push({ref_index[0], ref_index[1]});
    visited.insert({ref_index[0], ref_index[1]});

    int x_min = 0;
    int y_min = 0;
    int x_max = shape_[0] - 1;
    int y_max = shape_[1] - 1;

    // BFS 循环
    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();

        // 如果当前位置为 1，返回该点
        if (matrix(x, y) == 1) {
            std::cout << "tes t2" << std::endl;

            return Eigen::Vector2i(x, y);
        }

        // 单方向搜索（四邻域）
        std::vector<std::pair<int, int>> directions = {
            {0, ang_acc_},
            {ang_acc_, 0},
            {0, -ang_acc_},
            {-ang_acc_, 0}
        };

        for (auto [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;

            // 边界检查 + 未访问检查
            if (nx >= x_min && nx <= x_max &&
                ny >= y_min && ny <= y_max &&
                !visited.count({nx, ny})) {
                visited.insert({nx, ny});
                queue.push({nx, ny});
            }
        }
    }
    std::cout << "tes t3" << std::endl;

    return Eigen::Vector2i(-1, -1);
} 

Eigen::Vector2i FFTPlanner::FindNearestMargin(matrix &matrix, const Eigen::VectorXi &ref_idx, const Eigen::Vector2i &terminal_index,
                                    DType r_safe, DType r_extreme) {

    if ((ref_idx.array() == terminal_index.array()).all()) {
        // 两个向量完全相等时执行
        return ref_idx;
    }
    r_extreme = std::ceil(r_extreme);
    r_safe = std::round(r_safe);
    DType dis = std::max(static_cast<DType>((terminal_index.cast<DType>() - ref_idx.cast<DType>()).norm()), 1.0);
    DType kx = (terminal_index[0] - ref_idx[0]) / dis;
    DType ky = (terminal_index[1] - ref_idx[1]) / dis;

    DType x_min = std::max(ref_idx[0] - r_extreme, 0.0);
    DType x_max = std::min(ref_idx[0] + r_extreme, static_cast<DType>(shape_[0] - 1));
    DType y_min = std::max(ref_idx[1] - r_extreme, 0.0);
    DType y_max = std::min(ref_idx[1] + r_extreme, static_cast<DType>(shape_[1] - 1));
    DType times = std::round(dis / r_safe);
    auto ex_visited = ref_idx;
        for (int i = 1; i <= times; ++i) {
        int x = static_cast<int>(std::round(ref_idx[0] + i * kx * r_safe));
        int y = static_cast<int>(std::round(ref_idx[1] + i * ky * r_safe));

        // 到达终点
        if (x == static_cast<int>(terminal_index[0]) && y == static_cast<int>(terminal_index[1])) {
            return Eigen::Vector2i(x, y);
        }

        // 不安全或超出边界
        if (x < x_min || x > x_max || y < y_min || y > y_max || matrix(x, y) == 0) {
            return ex_visited;
        }

        ex_visited << x, y;  // 更新上一个安全点
    }
    std::cout << "ex visited " << ex_visited << std::endl;
    return ex_visited;
}

void FFTPlanner::FFTProcess(std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> &processed,
                            const Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> &image, const int j) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gray_image;
    gray_image = image.cwiseMax(0.0).cwiseMin(255.0 / dis_max_ * cutoff_[j]);

    gray_image.col(0).setZero();
    gray_image.row(0).setZero();

    complex_matrix freq1 = fft2(gray_image, Hshape_).cwiseProduct(H_D_[j]);
    matrix filtered_image = ifft2(freq1).cwiseAbs().block(0, 0, shape_[0], shape_[1]);

    double threshold = 255.0 / dis_max_ * cutoff_[j] - epsilon_;
    filtered_image = (filtered_image.array() >= threshold).cast<DType>();
    safe_[j] = filtered_image;

    if (j == 0) {
        filtered_image = filtered_image.cast<DType>().array() + 1;
        freq1 = fft2(filtered_image, Hshape_).cwiseProduct(H_S_[j]);

        filtered_image = ifft2(freq1).cwiseAbs().block(0, 0, shape_[0], shape_[1]);
        solver_.set_gradient(filtered_image);
        solver_.cal_gmax_gmin();
    } else if (j == 1) {
        freq1 = fft2(filtered_image, Hshape_).cwiseProduct(H_S_[j]);
        filtered_image = ifft2(freq1).cwiseAbs().block(0, 0, shape_[0], shape_[1]);
        matrix downsampled_image = Downsample(filtered_image, dtsep_);
        matrix downsampled_safe_area = Downsample(safe_[j], dtsep_);
        matrix downsampled = (downsampled_image.array() * downsampled_safe_area.array()).matrix();
        auto greater0 = Argrelextrema(downsampled, true, 0, 1);
        auto greater1 = Argrelextrema(downsampled, true, 1, 1);
        std::set<std::pair<int, int>> set_max0;
        for (int i = 0; i < greater0.rows(); ++i) {
            set_max0.insert({greater0(i, 0) * dtsep_, greater0(i, 1) * dtsep_});
        }
        std::set<std::pair<int, int>> set_max1;
        for (int i = 0; i < greater1.rows(); ++i) {
            set_max1.insert({greater1(i, 0) * dtsep_, greater1(i, 1) * dtsep_});
        }

        auto tmp = GetIntersectionMatrix(set_max0, set_max1);
        std::cout << "tmp " << tmp << std::endl;
        int N = tmp.cols();
        Eigen::Matrix<DType, 1, Eigen::Dynamic> values(1, N);
        for (int i = 0; i < N; ++i) {
            int row = tmp(0, i);
            int col = tmp(1, i);
            values(0, i) = filtered_image(row, col);
        }

        Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> stacked(3, N);
        stacked.block(0, 0, 2, N) = tmp.cast<DType>();
        stacked.row(2) = values;

        processed[j] = stacked.transpose(); // N x 3
        // std::cout << "processs j " << processed[j] << std::endl;
    }
}

Eigen::MatrixXi FFTPlanner::GetIntersectionMatrix(const std::set<std::pair<int, int>> &set_max0,
                                      const std::set<std::pair<int, int>> &set_max1) {
    // 1. 求交集
    std::set<std::pair<int, int>> intersection;
    for (const auto &p : set_max0) {
        if (set_max1.find(p) != set_max1.end()) {
            intersection.insert(p);
        }
    }

    // 2. 创建 2×N 矩阵
    int N = static_cast<int>(intersection.size());
    Eigen::MatrixXi tmp(2, N);

    // 3. 依次写入 (x, y)
    int i = 0;
    for (const auto &p : intersection) {
        tmp(0, i) = p.first;  // 对应 x
        tmp(1, i) = p.second; // 对应 y
        ++i;
    }

    return tmp; // 尺寸为 (2, N)
}

complex_matrix FFTPlanner::fft2(const matrix &gray_image, int Hshape) {
    // 创建输出矩阵
    complex_matrix output(Hshape, Hshape);

    // 将输入转为行优先矩阵，保证内存布局和 NumPy 一致
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gray_row = gray_image;

    // FFTW 输入输出缓冲区
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Hshape * Hshape);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Hshape * Hshape);

    // 填充输入（严格按行优先 zero-padding）
    for (int i = 0; i < Hshape; i++) {
        for (int j = 0; j < Hshape; j++) {
            if (i < gray_row.rows() && j < gray_row.cols()) {
                in[i * Hshape + j][0] = gray_row(i, j); // 实部
                in[i * Hshape + j][1] = 0.0;            // 虚部
            } else {
                in[i * Hshape + j][0] = 0.0;
                in[i * Hshape + j][1] = 0.0;
            }
        }
    }

    // 创建 FFT plan（2D 行优先）
    fftw_plan p = fftw_plan_dft_2d(Hshape, Hshape, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    // 将 FFT 结果写入 Eigen 矩阵
    for (int i = 0; i < Hshape; i++) {
        for (int j = 0; j < Hshape; j++) {
            output(i, j) = std::complex<double>(out[i * Hshape + j][0], out[i * Hshape + j][1]);
        }
    }

    // 清理
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return output;
}

complex_matrix FFTPlanner::ifft2(const complex_matrix &freq) {
    const int rows = freq.rows();
    const int cols = freq.cols();

    if (rows == 0 || cols == 0) {
        return complex_matrix();
    }

    complex_matrix output(rows, cols);

    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // 填充输入数据，取共轭（因为 ifft = fft(共轭)/N）
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = freq(i, j).real(); // 实部不变
            in[idx][1] = freq(i, j).imag(); // 虚部取负（共轭）
        }
    }

    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // 应用规范化因子
    const double scale = 1.0 / (rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            output(i, j) = std::complex<double>(out[idx][0] * scale, out[idx][1] * scale);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return output;
}

matrix FFTPlanner::Downsample(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, int dtsep) {
    int out_rows = (mat.rows() + dtsep - 1) / dtsep;
    int out_cols = (mat.cols() + dtsep - 1) / dtsep;

    matrix down(out_rows, out_cols);

    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            int src_i = i * dtsep;
            int src_j = j * dtsep;
            down(i, j) = mat(src_i, src_j);
        }
    }
    return down;
}

Eigen::MatrixXi FFTPlanner::Argrelextrema(const Eigen::MatrixXd &A, bool find_max, int axis, int order) {
    if (order < 1) order = 1;
    const int rows = (int)A.rows();
    const int cols = (int)A.cols();

    std::vector<std::pair<int,int>> hits;
    if (axis == 0) {
        // move along rows for each column
        if (rows <= 2*order) return Eigen::MatrixXi(0,2); // no valid index
        for (int j = 0; j < cols; ++j) {
            for (int i = order; i <= rows - order - 1; ++i) {
                double center = A(i,j);
                bool ok = true;
                for (int k = i - order; k <= i + order; ++k) {
                    if (k == i) continue;
                    double val = A(k, j);
                    if (find_max) {
                        if (!(center > val)) { ok = false; break; }
                    } else {
                        if (!(center < val)) { ok = false; break; }
                    }
                }
                if (ok) hits.emplace_back(i, j);
            }
        }
    } else if (axis == 1) {
        // move along cols for each row
        if (cols <= 2*order) return Eigen::MatrixXi(0,2);
        for (int i = 0; i < rows; ++i) {
            for (int j = order; j <= cols - order - 1; ++j) {
                double center = A(i,j);
                bool ok = true;
                for (int k = j - order; k <= j + order; ++k) {
                    if (k == j) continue;
                    double val = A(i, k);
                    if (find_max) {
                        if (!(center > val)) { ok = false; break; }
                    } else {
                        if (!(center < val)) { ok = false; break; }
                    }
                }
                if (ok) hits.emplace_back(i, j);
            }
        }
    } else {
        throw std::invalid_argument("axis must be 0 or 1");
    }

    Eigen::MatrixXi out((int)hits.size(), 2);
    for (int r = 0; r < (int)hits.size(); ++r) {
        out(r, 0) = hits[r].first;   // row
        out(r, 1) = hits[r].second;  // col
    }
    return out;
}

Eigen::MatrixXi FFTPlanner::ArgrelextremaGreater(const matrix &mat, int axis, int order) {
    std::vector<int> row_indices, col_indices;
    int rows = mat.rows();
    int cols = mat.cols();

    if (axis == 0) { // 沿列
        for (int j = 0; j < cols; ++j) {
            for (int i = order; i < rows - order; ++i) {
                bool is_max = true;
                for (int k = 1; k <= order; ++k) {
                    if (mat(i, j) <= mat(i - k, j) || mat(i, j) <= mat(i + k, j)) {
                        is_max = false;
                        break;
                    }
                }
                if (is_max) {
                    row_indices.push_back(i);
                    col_indices.push_back(j);
                }
            }
        }
    } else if (axis == 1) { // 沿行
        for (int i = 0; i < rows; ++i) {
            for (int j = order; j < cols - order; ++j) {
                bool is_max = true;
                for (int k = 1; k <= order; ++k) {
                    if (mat(i, j) <= mat(i, j - k) || mat(i, j) <= mat(i, j + k)) {
                        is_max = false;
                        break;
                    }
                }
                if (is_max) {
                    row_indices.push_back(i);
                    col_indices.push_back(j);
                }
            }
        }
    }

    Eigen::MatrixXi result(2, row_indices.size());
    for (size_t idx = 0; idx < row_indices.size(); ++idx) {
        result(0, idx) = row_indices[idx];
        result(1, idx) = col_indices[idx];
    }
    return result;
}

std::vector<std::pair<int, int>> FFTPlanner::LocalMaximaAxis0(const Eigen::MatrixXd &mat, int order) {
    std::vector<std::pair<int, int>> maxima;
    int rows = mat.rows();
    int cols = mat.cols();
    maxima.reserve(cols);
    for (int j = 0; j < cols; ++j) { // 列循环
        for (int i = order; i < rows - order; ++i) {
            bool is_max = true;
            for (int k = 1; k <= order; ++k) {
                if (mat(i, j) <= mat(i - k, j) || mat(i, j) <= mat(i + k, j)) {
                    is_max = false;
                    break;
                }
            }
            if (is_max) {
                maxima.emplace_back(i, j);
            }
        }
    }
    return maxima;
}

std::vector<std::pair<int, int>> FFTPlanner::LocalMaximaAxis1(const Eigen::MatrixXd &mat, int order) {
    std::vector<std::pair<int, int>> maxima;

    int rows = mat.rows();
    int cols = mat.cols();
    maxima.reserve(rows);
    for (int i = 0; i < rows; ++i) { // 行循环
        for (int j = order; j < cols - order; ++j) {
            bool is_max = true;
            for (int k = 1; k <= order; ++k) {
                if (mat(i, j) <= mat(i, j - k) || mat(i, j) <= mat(i, j + k)) {
                    is_max = false;
                    break;
                }
            }
            if (is_max) {
                maxima.emplace_back(i, j);
            }
        }
    }
    return maxima;
}

// void FFTPlanner::Reset() {
//   // TODO: reset safe_
//   // 重置 safe_：path_len_ 个 2D bool 矩阵（模拟 np.zeros）
//   // safe_.resize(path_len_);
//   // for (int i = 0; i < path_len_; ++i) {
//   //   safe_[i] = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(
//   //       shape_[0], shape_[1]);
//   // }

//   // 重置 rotation_ 为 identity
//   rotation_ = Eigen::Quaterniond::Identity();

//   // 向量重置
//   p0_ = Eigen::Vector3d::Zero();
//   linar_v_ = Eigen::Vector3d::Zero();
//   linar_a_ = Eigen::Vector3d::Zero();  // 加速度恒定为 0

//   target_direction_ = Eigen::Vector3d(1.0, 0.0, 0.0);
//   target_vector_ = Eigen::Vector3d(10.0, 0.0, 0.0);
//   target_ = Eigen::Vector3d(10.0, 0.0, 0.0);
//   target_distance_ = 0.0;

//   approaching_ = false;

//   last_end_point_ = Eigen::Vector3d::Zero();
//   last_end_val_ = Eigen::Vector3d::Zero();
//   last_position_ = Eigen::Vector3d::Zero();

//   // 时间差分相关变量
//   last_time_ = -1.0;  // 替代 Python 的 None
//   since_end_time_ = -1.0;
//   last_linear_velocity_ =
//       Eigen::Vector3d::Zero();  // 或用 bool 标志位记录是否初始化过
// }

// double FFTPlanner::sphericalDistance(double theta1, double phi1, double
// theta2,
//                                      double phi2) const {
//   double r = cutoff_[0];  // 对应 Python 中 self.cutoff[0]

//   double delta_phi = phi1 - phi2;
//   double cos_term = std::sin(theta1) * std::sin(theta2) * std::cos(delta_phi)
//   +
//                     std::cos(theta1) * std::cos(theta2);

//   double distance_sq = 2 * r * r * (1.0 - cos_term);

//   // 避免因浮点误差导致 sqrt 负值
//   return std::sqrt(std::max(distance_sq, 0.0));
// }

SVAResult FFTPlanner::CalSVA(const Eigen::Vector3d &alpha, const Eigen::Vector3d &beta, const Eigen::Vector3d &gamma,
                             const Eigen::Vector3d &a0, const Eigen::Vector3d &v0, const Eigen::Vector3d &p0,
                             double T) const {
    SVAResult out;
    const int ki = static_cast<int>(std::lround(T / kt_));
    std::cout << "ki " << ki << std::endl;
    out.s = 0.0;

    out.times = Eigen::VectorXd::Zero(ki + 1);
    out.tao = Eigen::VectorXd::Ones(ki + 1);

    out.p = Eigen::MatrixXd::Zero(ki + 1, 3);
    out.v = Eigen::MatrixXd::Zero(ki + 1, 3);
    out.a = Eigen::MatrixXd::Zero(ki + 1, 3);

    Eigen::VectorXd vl = Eigen::VectorXd::Zero(ki + 1);
    Eigen::VectorXd al = Eigen::VectorXd::Zero(ki + 1);

    out.p.row(0) = p0.transpose();
    out.v.row(0) = v0.transpose();
    out.a.row(0) = a0.transpose();

    vl[0] = v0.norm();
    al[0] = a0.norm();

    for (int i = 1; i <= ki; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(ki) * T;
        out.times[i] = t;

        out.p.row(i) = (alpha * (std::pow(t, 5) / 120.0) + 
                        beta * (std::pow(t, 4) / 24.0) +
                        gamma * (std::pow(t, 3) / 6.0) + 
                        a0 * (std::pow(t, 2) / 2.0) + v0 * t + p0).transpose();
        out.s += (out.p.row(i) - out.p.row(i - 1)).norm();

        out.v.row(i) = (alpha * (std::pow(t, 4) / 24.0) + 
                    beta * (std::pow(t, 3) / 6.0) + 
                    gamma * (std::pow(t, 2) / 2.0) +
                    a0 * t + v0).transpose();

        vl[i] = out.v.row(i).norm();

        if (vl[i] > max_spd_) {
            out.tao[i] = vl[i] / max_spd_;
        }

        out.a.row(i) = (alpha * (std::pow(t, 3) / 6.0) + 
                    beta * (std::pow(t, 2) / 2.0) + 
                    gamma * t + 
                    a0).transpose();

        al[i] = out.a.row(i).norm();
        if (al[i] > max_acc_) {
            out.tao[i] = std::max(out.tao[i], std::sqrt(al[i] / max_acc_));
        }
    }
    return out;
}

SVAResult FFTPlanner::CalSVAT(const Eigen::Vector3d& alpha,
                          const Eigen::Vector3d& beta,
                          const Eigen::Vector3d& gamma,
                          const Eigen::Vector3d& a0, const Eigen::Vector3d&
                          v0, const Eigen::Vector3d& p0, double t) {
    SVAResult out;
    out.p.resize(3, 1);
    out.v.resize(3, 1);
    out.a.resize(3, 1);
    // 预计算幂，减少对 std::pow 的重复调用
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double t4 = t2 * t2;
    const double t5 = t3 * t2;

    // 位置
    out.p(0) = alpha(0) * t5 / 120.0 + beta(0) * t4 / 24.0 + gamma(0) * t3 / 6.0 + a0(0) * t2 / 2.0 + v0(0) * t + p0(0);
    out.p(1) = alpha(1) * t5 / 120.0 + beta(1) * t4 / 24.0 + gamma(1) * t3 / 6.0 + a0(1) * t2 / 2.0 + v0(1) * t + p0(1);
    out.p(2) = alpha(2) * t5 / 120.0 + beta(2) * t4 / 24.0 + gamma(2) * t3 / 6.0 + a0(2) * t2 / 2.0 + v0(2) * t + p0(2);

    // 速度
    out.v(0) = alpha(0) * t4 / 24.0 + beta(0) * t3 / 6.0 + gamma(0) * t2 / 2.0 + a0(0) * t + v0(0);
    out.v(1) = alpha(1) * t4 / 24.0 + beta(1) * t3 / 6.0 + gamma(1) * t2 / 2.0 + a0(1) * t + v0(1);
    out.v(2) = alpha(2) * t4 / 24.0 + beta(2) * t3 / 6.0 + gamma(2) * t2 / 2.0 + a0(2) * t + v0(2);

    // 加速度
    out.a(0) = alpha(0) * t3 / 6.0 + beta(0) * t2 / 2.0 + gamma(0) * t + a0(0);
    out.a(1) = alpha(1) * t3 / 6.0 + beta(1) * t2 / 2.0 + gamma(1) * t + a0(1);
    out.a(2) = alpha(2) * t3 / 6.0 + beta(2) * t2 / 2.0 + gamma(2) * t + a0(2);

    return out;
}

PolyParams FFTPlanner::CalFullyDefinedParam(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& v0,
    const Eigen::Vector3d& a0, const Eigen::Vector3d& pf,
    const Eigen::Vector3d& vf1, const Eigen::Vector3d& af, double T) {
  PolyParams out;
  out.alpha.setZero();
  out.beta.setZero();
  out.gamma.setZero();

  const double T2 = T * T;
  const double T3 = T2 * T;
  const double T4 = T3 * T;
  const double T5 = T4 * T;

  Eigen::Matrix3d M;
  M << 720.0, -360.0 * T, 60.0 * T2, -360.0 * T, 168.0 * T2, -24.0 * T3,
      60.0 * T2, -24.0 * T3, 3.0 * T4;
  M /= T5;

  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d b;
    b[0] = pf[i] - p0[i] - v0[i] * T - 0.5 * a0[i] * T2;
    b[1] = vf1[i] - v0[i] - a0[i] * T;
    b[2] = af[i] - a0[i];

    Eigen::Vector3d y = M * b;
    out.alpha[i] = y[0];
    out.beta[i] = y[1];
    out.gamma[i] = y[2];
  }

  return out;
}

double FFTPlanner::CalFullyDefinedJ(const Eigen::Vector3d& alpha,
                                       const Eigen::Vector3d& beta,
                                       const Eigen::Vector3d& gamma, double
                                       T) {
  const double T2 = T * T;
  const double T3 = T2 * T;
  const double T4 = T3 * T;

  double Jsum = 0.0;
  for (int i = 0; i < 3; ++i) {
    double Ji = std::pow(gamma[i], 2) + beta[i] * gamma[i] * T +
                (1.0 / 3.0) * std::pow(beta[i], 2) * T2 +
                (1.0 / 3.0) * alpha[i] * gamma[i] * T2 +
                (1.0 / 4.0) * alpha[i] * beta[i] * T3 +
                (1.0 / 20.0) * std::pow(alpha[i], 2) * T4;
    Jsum += Ji;
  }
  return Jsum;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>
FFTPlanner::CalForwardKinematicsDivide(Eigen::MatrixXd pl, Eigen::MatrixXd vl, Eigen::MatrixXd al,
                                       Eigen::Vector3d p1, Eigen::VectorXd times, Eigen::VectorXd tao, double T1,
                                       double T2, bool p1_safe) {
    Eigen::Vector3d p0 = pl.row(0);
    Eigen::Vector3d v0 = vl.row(0);
    Eigen::Vector3d a0 = al.row(0);

    Eigen::VectorXd v1, a1, p2, v2, a2;
    Eigen::VectorXd times1, times2, tao1, tao2;

    int idx1 = -1;
    for (int i = 0; i < times.size(); ++i) {
        if (fabs(times[i] - T1) < 1e-6) {
            idx1 = i;
            break;
        }
    }

    v1 = vl.row(idx1);
    a1 = al.row(idx1);

    if (T2 > 0) {
        p2 = pl.row(pl.rows() - 1);
        v2 = vl.row(vl.rows() - 1);
        a2 = al.row(al.rows() - 1);        
    } else {
        v1 = Eigen::VectorXd::Zero(3);
        a1 = Eigen::VectorXd::Zero(3);
    }

    if (!p1_safe) {
        Eigen::Vector3d alpha1, beta1, gamma1;
        Eigen::Vector3d alpha2, beta2, gamma2;
        if (T2 > 0) {
            for (int i = 0; i < 3; ++i) {
                double K1 = 1.0;
                double K2 = -p0[i] - v0[i] * T1 - 0.5 * a0[i] * T1 * T1;
                double M1 = 0.0;
                double M2 = v1[i] - v0[i] - a0[i] * T1;

                double kalpha1 = (1.0 / std::pow(T1, 5)) * (320.0 * K1 - 120.0 * M1 * T1);
                double balpha1 = (1.0 / std::pow(T1, 5)) * (320.0 * K2 - 120.0 * M2 * T1);
                double kbeta1 = (1.0 / std::pow(T1, 4)) * (-200.0 * K1 + 72.0 * M1 * T1);
                double bbeta1 = (1.0 / std::pow(T1, 4)) * (-200.0 * K2 + 72.0 * M2 * T1);
                double kgamma1 = (1.0 / std::pow(T1, 3)) * (40.0 * K1 - 12.0 * M1 * T1);
                double bgamma1 = (1.0 / std::pow(T1, 3)) * (40.0 * K2 - 12.0 * M2 * T1);

                alpha1[i] = kalpha1 * p1[i] + balpha1;
                beta1[i] = kbeta1 * p1[i] + bbeta1;
                gamma1[i] = kgamma1 * p1[i] + bgamma1;

                if (T2 > 0) {
                    double A = std::pow(T1, 4) * kalpha1 / 24.0 + std::pow(T1, 3) * kbeta1 / 6.0 +
                               std::pow(T1, 2) * kgamma1 / 2.0;
                    double B = std::pow(T1, 4) * balpha1 / 24.0 + std::pow(T1, 3) * bbeta1 / 6.0 +
                               std::pow(T1, 2) * bgamma1 / 2.0 + a0[i] * T1 + v0[i];
                    double C = std::pow(T1, 3) * kalpha1 / 6.0 + std::pow(T1, 2) * kbeta1 / 2.0 + T1 * kgamma1;
                    double D = std::pow(T1, 3) * balpha1 / 6.0 + std::pow(T1, 2) * bbeta1 / 2.0 + T1 * bgamma1 + a0[i];

                    double K1 = -(1.0 + A * T2 + 0.5 * C * T2 * T2);
                    double K2 = p2[i] - B * T2 - 0.5 * D * T2 * T2;
                    double M1 = -(A + C * T2);
                    double M2 = v2[i] - B - D * T2;
                    double N1 = -C;
                    double N2 = -D;

                    double kalpha2 = (1.0 / std::pow(T2, 5)) * (720.0 * K1 - 360.0 * M1 * T2 + 60.0 * N1 * T2 * T2);
                    double balpha2 = (1.0 / std::pow(T2, 5)) * (720.0 * K2 - 360.0 * M2 * T2 + 60.0 * N2 * T2 * T2);
                    double kbeta2 = (1.0 / std::pow(T2, 4)) * (-360.0 * K1 + 168.0 * M1 * T2 - 24.0 * N1 * T2 * T2);
                    double bbeta2 = (1.0 / std::pow(T2, 4)) * (-360.0 * K2 + 168.0 * M2 * T2 - 24.0 * N2 * T2 * T2);
                    double kgamma2 = (1.0 / std::pow(T2, 3)) * (60.0 * K1 - 24.0 * M1 * T2 + 3.0 * N1 * T2 * T2);
                    double bgamma2 = (1.0 / std::pow(T2, 3)) * (60.0 * K2 - 24.0 * M2 * T2 + 3.0 * N2 * T2 * T2);

                    alpha2[i] = kalpha2 * p1[i] + balpha2;
                    beta2[i] = kbeta2 * p1[i] + bbeta2;
                    gamma2[i] = kgamma2 * p1[i] + bgamma2;
                }
            }
        } else {
            T1 = 2.0 * (p1 - p0).norm() / v0.norm() + kt_;
            auto param = CalFullyDefinedParam(p0, v0, a0, p1, v1, a1, T1);
            alpha1 = param.alpha;
            beta1 = param.beta;
            gamma1 = param.gamma;
            std::cout << "T1 " << T1 << std::endl;
            std::cout << "param alpha " << param.alpha << std::endl;
            std::cout << "param beta " << param.beta << std::endl;
            std::cout << "param gamma " << param.gamma << std::endl;

        }
        auto sva1 = CalSVA(alpha1, beta1, gamma1, a0, v0, p0, T1);
        Eigen::MatrixXd pl, vl, al;
        Eigen::VectorXd times, tao, tao2;
        if (T2 > 0) {
            auto sva2 = CalSVA(alpha2, beta2, gamma2, sva1.a.row(sva1.a.rows() - 1), sva1.v.row(sva1.v.rows() - 1), p1, T2);
            // pl = [pl1; pl2[1:]]
            pl.resize(sva1.p.rows() + sva2.p.rows() - 1, 3);
            pl.topRows(sva1.p.rows()) = sva1.p;
            pl.bottomRows(sva2.p.rows() - 1) = sva2.p.bottomRows(sva2.p.rows() - 1);

            // times = [times1; times2[1:] + T1]
            times.resize(sva1.times.size() + sva2.times.size() - 1);
            times.head(sva1.times.size()) = sva1.times;
            times.tail(sva2.times.size() - 1) = sva2.times.tail(sva2.times.size() - 1).array() + T1;

            // vl = [vl1; vl2[1:]]
            vl.resize(sva1.v.rows() + sva2.v.rows() - 1, 3);
            vl.topRows(sva1.v.rows()) = sva1.v;
            vl.bottomRows(sva2.v.rows() - 1) = sva2.v.bottomRows(sva2.v.rows() - 1);

            // al = [al1; al2[1:]]
            al.resize(sva1.a.rows() + sva2.a.rows() - 1, 3);
            al.topRows(sva1.a.rows()) = sva1.a;
            al.bottomRows(sva2.a.rows() - 1) = sva2.a.bottomRows(sva2.a.rows() - 1);

            // tao = [tao1; tao2[1:]]
            tao.resize(sva1.tao.size() + sva2.tao.size() - 1);
            tao.head(sva1.tao.size()) = sva1.tao;
            tao.tail(sva2.tao.size() - 1) = sva2.tao.tail(sva2.tao.size() - 1);

            // tao2 = tao2[1:]  // 不包括第一个截断点
            sva2.tao = sva2.tao.tail(sva2.tao.size() - 1);
        } else {
            pl = sva1.p;
            times = sva1.times;
            vl = sva1.v;
            al = sva1.a;
            tao = sva1.tao;
        }

        for (int i = 0; i < pl.rows(); ++i) {
            if ((pl.row(i) - p1.transpose()).cwiseAbs().maxCoeff() < 1e-9) {
                idx1 = i;
                break;
            }
        }

        p1 = pl.row(idx1);
        v1 = vl.row(idx1);
        a1 = al.row(idx1);
    } else {
        tao1 = tao.topRows(idx1 + 1);
        if (T2 > 0) {
            tao2 = tao.bottomRows(tao.size() - idx1 - 1);
        } else {
            tao2 = Eigen::VectorXd(); // 空向量（相当于 None）
        }
    }
    std::cout << "test tao1 " << tao1 << std::endl;
    std::cout << "test tao2 " << tao2 << std::endl;
    std::cout << "test iter " << iter_ << std::endl;
    for (int i = 0; i < iter_; ++i) {
        if (tao.maxCoeff() <= 1 + 1e-2) {
            break;
        }
        if (tao1.maxCoeff() > 1 + 1e-2) {
            auto tao_t = tao1.maxCoeff();
            int idx_t = -1;
            for (int i = 0; i < tao.size(); ++i) {
                if (std::abs(tao(i) - tao_t) < 1e-6) {
                    idx_t = i;
                    break; // 只取第一个满足条件的
                }
            }
            auto p_t = pl.row(idx_t);
            auto v_t = vl.row(idx_t) / tao_t;
            auto a_t = al.row(idx_t) / (tao_t * tao_t);
            if (idx_t == idx1) {
                v1 = v_t;
                a1 = a_t;
            }
            DType c_0t = (p_t - p0.transpose()).norm();
            auto vec_0t = p_t - p0.transpose();
            DType theta_0t = std::acos(v0.dot(vec_0t) / (v0.norm() * vec_0t.norm()));
            DType s0t = c_0t * theta_0t / (2.0 * sin(theta_0t / 2.0) + 1e-6);
            if (!(s0t > 0.0 && s0t < c_0t * 3.14)) {
                s0t = c_0t * 1.1;
            }
            DType T_0t = 2.0 *s0t / (v0.norm() + v_t.norm());
            auto param_t0 = CalFullyDefinedParam(p0, v0, a0, p_t, v_t, a_t, T_0t);
            auto sva_t0 = CalSVA(param_t0.alpha, param_t0.beta, param_t0.gamma, a0, v0, p0, T_0t);

            auto c_t1 = (p1 - p_t.transpose()).norm();
            auto vec_t1 = p1 - p_t.transpose();
            DType theta_t1 = std::acos(v_t.dot(vec_t1) / (v_t.norm() * vec_t1.norm()));
            DType st1 = c_t1 * theta_t1 / (2.0 * sin(theta_t1 / 2.0) + 1e-6);
            if (!(st1 > 0 && st1 < c_t1 * 3.14)) {
                st1 = c_t1 * 1.1;
            }
            DType T_t1 = 2.0 * st1 / (v_t.norm() + v1.norm());
            auto param_t1 = CalFullyDefinedParam(p_t, v_t, a_t, p1, v1, a1, T_t1);
            auto sva_t1 = CalSVA(param_t1.alpha, param_t1.beta, param_t1.gamma, a_t, v_t, p_t, T_t1);

            Eigen::MatrixXd pl1 = ConcatMatrix(sva_t0.p, sva_t1.p, 1);
            Eigen::VectorXd times1 = ConcatVector(sva_t0.times, sva_t1.times, 1, T_0t);
            Eigen::MatrixXd vl1 = ConcatMatrix(sva_t0.v, sva_t1.v, 1);
            Eigen::MatrixXd al1 = ConcatMatrix(sva_t0.a, sva_t1.a, 1);
            Eigen::VectorXd tao1 = ConcatVector(sva_t0.tao, sva_t1.tao, 1);

            DType delta_t = times1[times1.size() - 1] - times1[0] - T1;
            pl = ConcatMatrix(pl1, pl, idx1 + 1);
            times = ConcatVector(times1, times, idx1 + 1, delta_t);
            vl = ConcatMatrix(vl1, vl, idx1 + 1);
            al = ConcatMatrix(al1, al, idx1 + 1);
            tao = ConcatVector(tao1, tao, idx1 + 1);

            int idx1 = -1;
            for (int i = 0; i < pl.rows(); ++i) {
                bool match = true;
                for (int j = 0; j < pl.cols(); ++j) {
                    if (std::round(pl(i, j) * 100.0) / 100.0 != std::round(p1(j) * 100.0) / 100.0) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    idx1 = i;
                    break;
                }
            }
            double T1 = times1(times1.size() - 1);
            tao1 = tao.head(idx1 + 1); 
            if (T2 > 0 && idx1 + 1 < tao.size()) {
                tao2 = tao.tail(tao.size() - idx1 - 1);
            } else {
                tao2.resize(0);
            }
        }
        if (tao2.size() > 0 && tao2.maxCoeff() > 1 + 1e-2) {
            DType tao_t = tao2.maxCoeff();
            int idx_t = -1;
            for (int i = 0; i < tao.size(); ++i) {
                if (std::abs(tao(i) - tao_t) < 1e-6) {
                    idx_t = i;
                    break;
                }
            }
            auto p_t = pl.row(idx_t);
            auto v_t = vl.row(idx_t) / tao_t;
            auto a_t = al.row(idx_t) / (tao_t * tao_t);

            DType c_1t = (p_t - p1.transpose()).norm();
            auto vec_1t = p_t - p1.transpose();
            auto theta_1t = acos(v1.dot(vec_1t) / (v1.norm() * vec_1t.norm()));
            DType s1t = c_1t * theta_1t / (2.0 * sin(theta_1t / 2.0) + 1e-6);

            if (!(s1t > 0.0 && s1t < c_1t * 3.14)) {
                s1t = c_1t *1.1;
            }
            DType T_1t = 2.0 * s1t / (v1.norm() + v_t.norm());

            auto param1 = CalFullyDefinedParam(p1, v1, a1, p_t, v_t, a_t, T_1t);
            auto sva1 = CalSVA(param1.alpha, param1.beta, param1.gamma, a1, v1, p1, T_1t);

            DType c_t2 = (p_t - p2.transpose()).norm();
            auto vec_t2 = p2 - p_t.transpose();
            DType theta_t2 = acos(v_t.dot(vec_t2) / (v_t.norm() * vec_t2.norm()));
            DType st2 = c_t2 * theta_t2 / (2.0 * sin(theta_t2 / 2.0) + 1e-6);
            if (!(st2 > 0 && st2 < c_t2 * 3.14)) {
                st2 = c_t2 * 1.1;
            }

            DType T_t2 = 2.0 * st2 / (v_t.norm() * v2.norm());
            auto param2 = CalFullyDefinedParam(p_t, v_t, a_t, p2, v2, a2, T_t2);
            auto sva2 = CalSVA(param2.alpha, param2.beta, param2.gamma, a_t, v_t, p_t, T_t2);

            Eigen::MatrixXd pl2 = ConcatMatrix(sva1.p, sva2.p, 1);
            Eigen::VectorXd times2 = ConcatVector(sva1.times, sva2.times, 1, T_1t);
            Eigen::MatrixXd vl2 = ConcatMatrix(sva1.v, sva2.v, 1);
            Eigen::MatrixXd al2 = ConcatMatrix(sva1.a, sva2.a, 1);
            Eigen::VectorXd tao2 = ConcatVector(sva1.tao, sva2.tao, 1);

            pl = ConcatMatrix(pl.topRows(idx1), pl2);
            times = ConcatVector(times.head(idx1), times2, 0, T1);
            vl = ConcatMatrix(vl.topRows(idx1), vl2);
            al = ConcatMatrix(al.topRows(idx1), al2);
            tao = ConcatVector(tao.head(idx1), tao2);

            int idx1 = -1;
            for (int i = 0; i < pl.rows(); ++i) {
                bool match = true;
                for (int j = 0; j < pl.cols(); ++j) {
                    if (std::round(pl(i, j) * 100.0) / 100.0 != std::round(p1(j) * 100.0) / 100.0) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    idx1 = i;
                    break;
                }
            }
            double T2 = times2(times2.size() - 1) - times2(0);
            tao1 = tao.head(idx1 + 1);
            if (T2 > 0 && idx1 + 1 < tao.size()) {
                tao2 = tao.tail(tao.size() - idx1 - 1);
            } else {
                tao2.resize(0);
            }
        }
    }
    std::cout << "pl " << pl << std::endl;
    std::cout << "vl " << vl << std::endl;
    std::cout << "al " << al << std::endl;
    std::cout << "times " << times << std::endl;


    return std::make_tuple(pl, vl, al, times);
}

BestIdxEndResult FFTPlanner::CalBestIdxEnd(const Eigen::Vector3d &p0, const Eigen::Vector3d &v0,
                                           const Eigen::Vector3d &a0, const Eigen::Vector3d &p2,
                                           Eigen::Vector3d v2) {
    BestIdxEndResult ret;
    DType T1, T2;
    DType v2_z = v2[2];
    DType s_all = 0.0;
    matrix tao1, tao2;
    DType delta_dis, s1, s2;
    Eigen::MatrixXd e_p1, p_sol, p_sol_ex;
    Eigen::Vector3d v1;
    Eigen::VectorXd a_sol;
    DType a1, b1, a2, b2;
    Eigen::MatrixXd pl, vl, al;
    Eigen::VectorXd tao, times;
    for (int i = 0; i < iter_; ++i) {
        if (i == 0) {
           if (((p2.array() + 1.0).abs() < 1e-6).any()) {
                DType c1 = cutoff_[0];
                auto ep1 = v0.normalized();
                DType theta1 = std::acos(v0.dot(ep1)) / (v0.norm() * ep1.norm()); 
                s1 = c1 * theta1 / (2.0 * std::sin(theta1 / 2.0) + 1e-6);
                if (!(s1 > 0.0 && s1 < c1 * 3.1415926)) {
                    s1 = c1 * 1.1;
                }

                T1 = 2.0 * s1 / v0.norm() + kt_;
                T2 = 0.0;
                s2 = 0.0;
                // TODO: tao2
                Eigen::Vector3d v1 = Eigen::Vector3d::Zero();
            } else {
                if (approaching_) {
                    DType c1 = cutoff_[0];
                    auto vec1 = p2 - p0;
                    DType theta1 = std::acos(v0.dot(vec1) / (v0.norm() * vec1.norm()));
                    DType s1 = c1 * theta1 / (2.0 * std::sin(theta1 / 2.0) + 1e-6);
                    if (!(s1 > 0.0 && s1 < c1 * 3.1415826)) {
                        s1 = c1 * 1.1;
                    }
                    DType s2 = std::abs(target_distance_ * cutoff_[0]);
                    DType V1 = std::sqrt(2 * max_acc_ * s2);
                    T2 = 2.0 * s2 * V1;
                    T1 = 2.0 * s1 / (V1 + v0.norm());
                }
                else {
                    DType c1 = cutoff_[0];
                    DType c2 = cutoff_[1] - cutoff_[0];
                    auto vec1 = p2 - p0;
                    DType theta1 = std::acos(v0.dot(vec1) / (v0.norm() * vec1.norm())) / 2.0;
                    DType s1 = c1 * theta1 / (2.0 * std::sin(theta1 / 2.0) + 1e-6);
                    if (!(s1 > 0.0 && s1 < c1 * 3.1415926)) {
                        s1 = c1  * 1.1;
                    }
                    DType theta2 = theta1;
                    DType s2 = c2 * theta2 / (2.0 * std::sin(theta2 / 2.0) + 1e-6);
                    if (!(s2 > 0.0 && s2 < c2 * 3.1415926)) {
                        s2 = c2  * 1.1;
                    }
                    T1 = 2.0 * s1 / (exp_spd_ + v0.norm());
                    T2 = s2 / exp_spd_;
                }

                auto param = CalFullyDefinedParam(Eigen::Vector3d::Zero(), linar_v_, linar_a_, p2, v2,
                                                  Eigen::Vector3d::Zero(), T1 + T2);
                auto sva = CalSVAT(param.alpha, param.beta, param.gamma, linar_a_, linar_v_, Eigen::Vector3d::Zero(), T1);
                e_p1 = sva.p.normalized();

                v1 = ComputeOptimalV1(p0, v0, a0, sva.p, p2, v2, Eigen::Vector3d::Zero(), T1, T2);
            }
        }
        else {
            if ((tao.size() > 0 && tao.maxCoeff()) < 1.1 || delta_dis < cutoff_[0] * angle_ptx_ / std::sqrt(2)) {
                break;
            }
            if (((p2.array() + 1.0).abs() < 1e-6).any()) {
                T1 = 2.0 * s1 / v0.norm() + kt_;
                T2 = 0.0;
                s2 = 0.0;
                // TODO: tao2 = none
                e_p1 = p_sol.normalized();
            } else {
                e_p1 = p_sol.normalized();
                v2 = (15.0 * (p2 - p_sol) - 7.0 * T2 * v1 - std::pow(T2, 2) * a_sol) / (8.0 * T2);
                v2[2] = v2_z;
                DType T1_ex = T1;
                DType T2_ex = T2;
                DType V0 = v0.norm();
                DType V2 = v2.norm();
                if (tao2.maxCoeff() > 1.0 + 1e-2) {
                    v2 /= std::max(tao2.maxCoeff(), 1.1);
                }
                DType V1 = v1.norm();
                if (tao1.maxCoeff() > 1.0 + 1e-2) {
                    V1 /= std::max(tao1.maxCoeff(), 1.1);
                }
                T1 = 2.0 * s1 / (V1 + V0);
                T2 = 2.0 * s2 / (V1 + V2);
                v1 = ComputeOptimalV1(p0, v0, a0, p_sol, p2, v2, Eigen::Vector3d::Zero(), T1, T2);
                if (T1 > 3.1415926 * V1 / max_acc_ || T2 > 3.1415926 * V1 / max_acc_ || target_vector_.dot(v1) < 0) {
                    T1 = T1_ex;
                    T2 = T2_ex;
                    break;
                }
            }
        }
        Eigen::Vector3d a, b;

        // param_* 是三维数组 (2, 3, 2)
        Eigen::Tensor<double, 3> param_alpha(2, 3, 2);
        Eigen::Tensor<double, 3> param_beta(2, 3, 2);
        Eigen::Tensor<double, 3> param_gamma(2, 3, 2);

        // 初始化为 0
        param_alpha.setZero();
        param_beta.setZero();
        param_gamma.setZero();

        for (int idx = 0; idx < 3; ++idx) {
            DType K1 = 1;
            DType K2 = -p0(idx) - v0(idx) * T1 - 0.5 * a0(idx) * T1 * T1;
            DType M1 = 0.0;
            DType M2 = v1(idx) - v0(idx) - a0(idx) * T1;

            DType kalpha1 = (1.0 / std::pow(T1, 5)) * (320.0 * K1 - 120.0 * M1 * T1);
            DType balpha1 = (1.0 / std::pow(T1, 5)) * (320.0 * K2 - 120.0 * M2 * T1);
            DType kbeta1 = (1.0 / std::pow(T1, 4)) * (-200.0 * K1 + 72.0 * M1 * T1);
            DType bbeta1 = (1.0 / std::pow(T1, 4)) * (-200.0 * K2 + 72.0 * M2 * T1);
            DType kgamma1 = (1.0 / std::pow(T1, 3)) * (40.0 * K1 - 12.0 * M1 * T1);
            DType bgamma1 = (1.0 / std::pow(T1, 3)) * (40.0 * K2 - 12.0 * M2 * T1);

            // 赋值
            param_alpha(0, idx, 0) = kalpha1;
            param_alpha(0, idx, 1) = balpha1;
            param_beta(0, idx, 0) = kbeta1;
            param_beta(0, idx, 1) = bbeta1;
            param_gamma(0, idx, 0) = kgamma1;
            param_gamma(0, idx, 1) = bgamma1;
            std::cout << "prepare kalpha1 " << kalpha1 << std::endl;
            std::cout << "prepare balpha1 " << balpha1 << std::endl;
            std::cout << "prepare kbeta1 " << kbeta1 << std::endl;
            std::cout << "prepare bbeta1 " << bbeta1 << std::endl;
            std::cout << "prepare kgamma1 " << kgamma1 << std::endl;
            std::cout << "prepare bgamma1 " << bgamma1 << std::endl;

            auto ab = CalAB(kalpha1, kbeta1, kgamma1, balpha1, bbeta1, bgamma1, T1);
            a1 = ab.first;
            b1 = ab.second;
            std::cout << "prepare a1 " << a1 << std::endl;
            std::cout << "prepare b1 " << b1 << std::endl;

            if (((p2.array() + 1.0).abs() < 1e-6).any()) {
                a2 = 0.0;
                b2 = 0.0;
            } else {
                // 计算 A, B, C, D
                double A =
                    std::pow(T1, 4) * kalpha1 / 24.0 + std::pow(T1, 3) * kbeta1 / 6.0 + std::pow(T1, 2) * kgamma1 / 2.0;
                double B = std::pow(T1, 4) * balpha1 / 24.0 + std::pow(T1, 3) * bbeta1 / 6.0 +
                           std::pow(T1, 2) * bgamma1 / 2.0 + a0[idx] * T1 + v0[idx];
                double C = std::pow(T1, 3) * kalpha1 / 6.0 + std::pow(T1, 2) * kbeta1 / 2.0 + T1 * kgamma1;
                double D = std::pow(T1, 3) * balpha1 / 6.0 + std::pow(T1, 2) * bbeta1 / 2.0 + T1 * bgamma1 + a0[idx];

                // std::cout << "prepare A " << A << std::endl;
                // std::cout << "prepare B " << B << std::endl;
                // std::cout << "prepare C " << C << std::endl;
                // std::cout << "prepare D " << D << std::endl;

                // p2, v2固定，a2限制为0
                double K1 = -(1.0 + A * T2 + 0.5 * C * std::pow(T2, 2));
                double K2 = p2[idx] - B * T2 - 0.5 * D * std::pow(T2, 2);
                double M1 = -(A + C * T2);
                double M2 = v2[idx] - B - D * T2;
                double N1 = -C;
                double N2 = -D;

                // std::cout << "prepare K1 " << K1 << std::endl;
                // std::cout << "prepare K2 " << K2 << std::endl;
                // std::cout << "prepare M1 " << M1 << std::endl;
                // std::cout << "prepare M2 " << M2 << std::endl;
                // std::cout << "prepare N1 " << N1 << std::endl;
                // std::cout << "prepare N2 " << N2 << std::endl;


                // 计算第二段参数
                double kalpha2 = (1.0 / std::pow(T2, 5)) * (720.0 * K1 - 360.0 * M1 * T2 + 60.0 * N1 * std::pow(T2, 2));
                double balpha2 = (1.0 / std::pow(T2, 5)) * (720.0 * K2 - 360.0 * M2 * T2 + 60.0 * N2 * std::pow(T2, 2));
                double kbeta2 = (1.0 / std::pow(T2, 4)) * (-360.0 * K1 + 168.0 * M1 * T2 - 24.0 * N1 * std::pow(T2, 2));
                double bbeta2 = (1.0 / std::pow(T2, 4)) * (-360.0 * K2 + 168.0 * M2 * T2 - 24.0 * N2 * std::pow(T2, 2));
                double kgamma2 = (1.0 / std::pow(T2, 3)) * (60.0 * K1 - 24.0 * M1 * T2 + 3.0 * N1 * std::pow(T2, 2));
                double bgamma2 = (1.0 / std::pow(T2, 3)) * (60.0 * K2 - 24.0 * M2 * T2 + 3.0 * N2 * std::pow(T2, 2));

                std::cout << "prepare kalpha2 " << kalpha2 << std::endl;
                std::cout << "prepare balpha2 " << balpha2 << std::endl;
                std::cout << "prepare kbeta2 " << kbeta2 << std::endl;
                std::cout << "prepare bbeta2 " << bbeta2 << std::endl;
                std::cout << "prepare kgamma2 " << kgamma2 << std::endl;
                std::cout << "prepare bgamma2 " << bgamma2 << std::endl;


                // 赋值给第二段 Tensor
                param_alpha(1, idx, 0) = kalpha2;
                param_alpha(1, idx, 1) = balpha2;
                param_beta(1, idx, 0) = kbeta2;
                param_beta(1, idx, 1) = bbeta2;
                param_gamma(1, idx, 0) = kgamma2;
                param_gamma(1, idx, 1) = bgamma2;
                
                auto ab = CalAB(kalpha2, kbeta2, kgamma2, balpha2, bbeta2, bgamma2, T2);
                a2 = ab.first;
                b2 = ab.second;
            }
            std::cout << idx << "param_alpha " << param_alpha << std::endl;
            std::cout << idx << "param_beta " << param_beta << std::endl;
            std::cout << idx << "param_gamma " << param_gamma << std::endl;

            a(idx) = a1 + a2;
            b(idx) = b1 + b2;
        }

        Eigen::Vector3d initial_p;
        initial_p << std::acos(e_p1(2, 0)), std::atan2(e_p1(1, 0), e_p1(0, 0)), 0.0;
        // std::cout << "prepare initial p " << initial_p << std::endl;
        // std::cout << "prepare p0 " << p0 << std::endl;
        // std::cout << "a " << a << std::endl;
        // std::cout << "b " << b << std::endl;
        // std::cout << "e p1 " << e_p1.cols() << " " << e_p1.rows() << " " << e_p1 << std::endl;
        solver_.set_p0(p0);
        solver_.set_a(a);
        solver_.set_b(b);
        solver_.set_safe(safe_[0]);
        solver_.set_rotation(rotation_);
        auto res = solver_.Solve(initial_p);
        int mx = res.first;
        int my = res.second;
        std::cout << "test solver " << mx << " " << my << std::endl;
        DType angle_horizontal = angle_ptx_ * (mx_centre_[1] - my);            // 水平角（左为正）
        DType angle_vertical = M_PI / 2.0 - angle_ptx_ * (mx_centre_[0] - mx); // 垂直角

        DType x_sol = cutoff_[0] * std::sin(angle_vertical) * std::cos(angle_horizontal);
        DType y_sol = cutoff_[0] * std::sin(angle_vertical) * std::sin(angle_horizontal);
        DType z_sol = cutoff_[0] * std::cos(angle_vertical);

        Eigen::Vector3d local_point(x_sol, y_sol, z_sol);  // 局部坐标
        p_sol = rotation_ * local_point;
        std::cout << "p sol " << p_sol << std::endl;
        if (i > 0) {
            delta_dis = (p_sol - p_sol_ex).norm();
        } else {
            delta_dis = 1.0;
        }
        p_sol_ex = p_sol;

        std::cout << "param alpha " << param_alpha << std::endl;
        std::cout << "param beta " << param_beta << std::endl;
        std::cout << "param gamma " << param_gamma << std::endl;

        Eigen::MatrixXd param_alpha0(3, 2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                param_alpha0(i, j) = param_alpha(0, i, j);

        auto alpha1 = param_alpha0.col(0).cwiseProduct(p_sol.col(0)) + param_alpha0.col(1);
        Eigen::MatrixXd param_beta0(3, 2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                param_beta0(i, j) = param_beta(0, i, j);
        auto beta1 = param_beta0.col(0).cwiseProduct(p_sol.col(0)) + param_beta0.col(1);
        Eigen::MatrixXd param_gamma0(3, 2);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                param_gamma0(i, j) = param_gamma(0, i, j);
        auto gamma1 = param_gamma0.col(0).cwiseProduct(p_sol.col(0)) + param_gamma0.col(1);
        std::cout << "alpha 1 " << alpha1 << std::endl;
        std::cout << "beta 1 " << beta1 << std::endl;
        std::cout << "gamma 1 " << gamma1 << std::endl;

        auto sva_res = CalSVA(alpha1, beta1, gamma1, a0, v0, p0, T1);
        a_sol = sva_res.a.row(sva_res.a.rows() - 1);
        std::cout << "a sol " << a_sol << std::endl;
        if (!(p2.array() == -1).any()) {
            Eigen::MatrixXd param_alpha1(3, 2);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j)
                    param_alpha1(i, j) = param_alpha(1, i, j);
            std::cout << "param alpha1 " << param_alpha1 << std::endl;
            auto alpha2 = param_alpha1.col(0).cwiseProduct(p_sol.col(0)) + param_alpha1.col(1);
            Eigen::MatrixXd param_beta1(3, 2);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j)
                    param_beta1(i, j) = param_beta(1, i, j);
            auto beta2 = param_beta1.col(0).cwiseProduct(p_sol.col(0)) + param_beta1.col(1);
            Eigen::MatrixXd param_gamma1(3, 2);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 2; ++j)
                    param_gamma1(i, j) = param_gamma(1, i, j);
            auto gamma2 = param_gamma1.col(0).cwiseProduct(p_sol.col(0)) + param_gamma1.col(1);
            auto sva_res2 = CalSVA(alpha2, beta2, gamma2, sva_res.a.row(sva_res.a.rows() - 1), 
            sva_res.v.row(sva_res.v.rows() - 1), p_sol, T2);

            pl.resize(sva_res.p.rows() + sva_res2.p.rows() - 1, 3);
            pl.topRows(sva_res.p.rows()) = sva_res.p;
            pl.bottomRows(sva_res2.p.rows() - 1) = sva_res2.p.bottomRows(sva_res2.p.rows() - 1);

            // times.resize(sva_res.times.rows() + sva_res2.times.rows() - 1, 3);
            // times.topRows(sva_res.times.rows()) = sva_res.times;
            // times.bottomRows(sva_res2.times.rows() - 1) = sva_res2.times.bottomRows(sva_res2.times.rows() - 1);

            times.resize(sva_res.times.size() + sva_res2.times.size() - 1);
            times.head(sva_res.times.size()) = sva_res.times;
            times.tail(sva_res2.times.size() - 1) = sva_res2.times.tail(sva_res2.times.size() - 1).array() + T1;

            vl.resize(sva_res.v.rows() + sva_res2.v.rows() - 1, 3);
            vl.topRows(sva_res.v.rows()) = sva_res.v;
            vl.bottomRows(sva_res2.v.rows() - 1) = sva_res2.v.bottomRows(sva_res2.v.rows() - 1);

            al.resize(sva_res.a.rows() + sva_res2.a.rows() - 1, 3);
            al.topRows(sva_res.a.rows()) = sva_res.a;
            al.bottomRows(sva_res2.a.rows() - 1) = sva_res2.a.bottomRows(sva_res2.a.rows() - 1);

            tao.resize(sva_res.tao.size() + sva_res2.tao.size() - 1);
            tao.head(sva_res.tao.size()) = sva_res.tao;
            tao.tail(sva_res2.tao.size() - 1) = sva_res2.tao.tail(sva_res2.tao.size() - 1);

            s_all = sva_res.s + sva_res2.s;
        } else {
            pl = sva_res.p;
            times = sva_res.times;
            vl = sva_res.v;
            al = sva_res.a;
            tao = sva_res.tao;
            s_all = sva_res.s;
        }
        std::cout << "test iter " << i << std::endl;
    }
 
    ret.p_sol = p_sol;
    ret.pl = pl;
    ret.vl = vl;
    ret.al = al;
    ret.times = times;
    ret.T1 = T1;
    ret.T2 = T2;
    ret.v1 = v1;
    ret.tao = tao;
    ret.s_all = s_all;
    return ret;
}

std::pair<double, double> FFTPlanner::CaLAB(double kalpha, double kbeta, double kgamma,
                                 double balpha, double bbeta, double bgamma,
                                 double T) {
    // 计算 a
    double a = kgamma*kgamma
             + T * kbeta * kgamma
             + (1.0/3.0) * (T*T) * kalpha * kgamma
             + (1.0/4.0) * (T*T*T) * kalpha * kbeta
             + (1.0/20.0) * (T*T*T*T) * kalpha * kalpha
             + (1.0/3.0) * (T*T) * kbeta * kbeta;

    // 计算 b
    double b = 2.0 * kgamma * bgamma
             + T * kbeta * bgamma
             + T * kgamma * bbeta
             + (1.0/3.0) * (T*T) * 2.0 * kbeta * bbeta
             + (1.0/3.0) * (T*T) * kalpha * bgamma
             + (1.0/3.0) * (T*T) * kgamma * balpha
             + (1.0/4.0) * (T*T*T) * kalpha * bbeta
             + (1.0/4.0) * (T*T*T) * kbeta * balpha
             + (1.0/20.0) * (T*T*T*T) * 2.0 * kalpha * balpha;

    return {a, b};
}

Eigen::Vector3d FFTPlanner::ComputeOptimalV1(const Eigen::Vector3d &p0, const Eigen::Vector3d &v0,
                                               const Eigen::Vector3d &a0, const Eigen::Vector3d &p1,
                                               const Eigen::Vector3d &p2, const Eigen::Vector3d &v2,
                                               const Eigen::Vector3d &a2, double T1, double T2) {
  double M = 4.0 * (std::pow(T2, 3) - std::pow(T1, 3)) /
             (T1 * T2 * (std::pow(T1, 2) + std::pow(T2, 2)));

  Eigen::Vector3d N =
      (std::pow(T2, 4) * (120.0 * (p1 - p0) - 48.0 * v0 * T1 - 6.0 * a0 * std::pow(T1, 2)) -
       std::pow(T1, 4) * (120.0 * (p2 - p1) - 48.0 * v2 * T2 + 6.0 * a2 * std::pow(T2, 2))) /
      (-18.0 * std::pow(T1, 2) * std::pow(T2, 2) * (std::pow(T1, 2) + std::pow(T2, 2)));

  double P = 3.0 * T1 *
             (std::pow(T2, 5) - 5.0 * std::pow(T1, 5) + 4.0 * std::pow(T1, 3) * std::pow(T2, 2)) /
             (16.0 * T2 * (std::pow(T1, 4) + std::pow(T2, 4)));

  Eigen::Vector3d Q =
      ((30.0 * (p2 - p1) - 14.0 * v2 * T2 + 2.0 * a2 * std::pow(T2, 2)) * std::pow(T1, 5) +
       (30.0 * (p1 - p0) - 14.0 * v0 * T1 - 2.0 * a0 * std::pow(T1, 2)) * std::pow(T2, 5)) /
      (16.0 * T1 * T2 * (std::pow(T1, 4) + std::pow(T2, 4)));

  return (P * N + Q) / (1.0 - P * M);
}

// Eigen::Vector2i FFTPlanner::findNearestOne(
//     const Eigen::Ref<const Eigen::MatrixXi>& matrix,
//     const Eigen::Vector2i& ref_index,
//     const Eigen::Vector2i& terminal_index) const {
//   const int rows = matrix.rows();
//   const int cols = matrix.cols();

//   // 如果原点就是安全点，直接返回
//   if (matrix(ref_index.x(), ref_index.y()) == 1) {
//     return ref_index;
//   }

//   // 访问标记
//   std::vector<std::vector<char>> visited(rows, std::vector<char>(cols, 0));
//   std::deque<Eigen::Vector2i> queue;

//   queue.push_back(ref_index);
//   visited[ref_index.x()][ref_index.y()] = 1;

//   int kx = (terminal_index.x() - ref_index.x()) >= 0 ? 1 : -1;
//   int ky = (terminal_index.y() - ref_index.y()) >= 0 ? 1 : -1;

//   int x_min = std::min(terminal_index.x(), ref_index.x());
//   int x_max = std::max(terminal_index.x(), ref_index.x());
//   int y_min = std::min(terminal_index.y(), ref_index.y());
//   int y_max = std::max(terminal_index.y(), ref_index.y());

//   while (!queue.empty()) {
//     Eigen::Vector2i cur = queue.front();
//     queue.pop_front();

//     int x = cur.x();
//     int y = cur.y();

//     // 若当前位置是 1，则返回
//     if (matrix(x, y) == 1) {
//       return cur;
//     }

//     // 单方向搜索： (0, ky*ang_acc) 与 (kx*ang_acc, 0)
//     const int dxs[2] = {0, kx * ang_acc};
//     const int dys[2] = {ky * ang_acc, 0};

//     for (int i = 0; i < 2; ++i) {
//       int nx = x + dxs[i];
//       int ny = y + dys[i];

//       // 限定在矩阵范围 & 转向安全区域内 & 未访问
//       if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && y_min <= ny &&
//           ny <= y_max && x_min <= nx && nx <= x_max && !visited[nx][ny]) {
//         visited[nx][ny] = 1;
//         queue.emplace_back(nx, ny);
//       }
//     }
//   }

//   return terminal_index;
// }

// Eigen::Vector2i FFTPlanner::findNearestOneOrigin(const Eigen::Ref<const
// Eigen::MatrixXi>& matrix,
//                                          Eigen::Vector2i ref_index) const
//     {
//         // 如果有 -1，按原逻辑直接返回 [-1, -1]
//         if (ref_index.x() == -1 || ref_index.y() == -1) {
//             return Eigen::Vector2i(-1, -1);
//         }

//         const int rows = matrix.rows();
//         const int cols = matrix.cols();

//         // map_to_edge(self.shape, ref_index) -> 夹到边界
//         ref_index.x() = std::clamp(ref_index.x(), 0, rows - 1);
//         ref_index.y() = std::clamp(ref_index.y(), 0, cols - 1);

//         // 如果原点就是安全点
//         if (matrix(ref_index.x(), ref_index.y()) == 1) {
//             return ref_index;
//         }

//         // 访问标记
//         std::vector<std::vector<char>> visited(rows, std::vector<char>(cols,
//         0)); std::deque<Eigen::Vector2i> queue;

//         queue.push_back(ref_index);
//         visited[ref_index.x()][ref_index.y()] = 1;

//         const int x_min = 0;
//         const int x_max = rows - 1;
//         const int y_min = 0;
//         const int y_max = cols - 1;

//         while (!queue.empty()) {
//             Eigen::Vector2i cur = queue.front();
//             queue.pop_front();

//             int x = cur.x();
//             int y = cur.y();

//             if (matrix(x, y) == 1) {
//                 return cur;
//             }

//             // 4-邻域，步长为 ang_acc
//             const int dxs[4] = { 0,  ang_acc,  0, -ang_acc };
//             const int dys[4] = { ang_acc, 0, -ang_acc,  0 };

//             for (int i = 0; i < 4; ++i) {
//                 int nx = x + dxs[i];
//                 int ny = y + dys[i];

//                 if (nx >= x_min && nx <= x_max &&
//                     ny >= y_min && ny <= y_max &&
//                     !visited[nx][ny])
//                 {
//                     visited[nx][ny] = 1;
//                     queue.emplace_back(nx, ny);
//                 }
//             }
//         }

//         return Eigen::Vector2i(-1, -1);
//     }

} // namespace FFTPlanner