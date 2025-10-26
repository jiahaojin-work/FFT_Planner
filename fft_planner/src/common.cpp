#include "common.h"

#include <iomanip>

namespace FFTPlanner {

Eigen::Vector2i Map2Edge(const Eigen::Vector2i& matrixShape,
                         const Eigen::Vector2i& index) {
  int rows = matrixShape.x();
  int cols = matrixShape.y();
  int x = index.x();
  int y = index.y();

  // 计算矩阵中心
  DType center_x = rows * 0.5;
  DType center_y = cols * 0.5;

  // 如果角标在矩阵范围内，直接返回
  if (x >= 0 && x < rows && y >= 0 && y < cols) {
    return index;
  }

  // 方向向量
  DType dir_x = x - center_x;
  DType dir_y = y - center_y;

  // 计算比例因子
  DType scale_x = std::numeric_limits<DType>::infinity();
  DType scale_y = std::numeric_limits<DType>::infinity();

  if (dir_x != 0) {
    scale_x =
        (dir_x < 0) ? (0 - center_x) / dir_x : (rows - 1 - center_x) / dir_x;
  }

  if (dir_y != 0) {
    scale_y =
        (dir_y < 0) ? (0 - center_y) / dir_y : (cols - 1 - center_y) / dir_y;
  }

  // 取最小比例因子
  DType scale = std::min(scale_x, scale_y);

  // 映射到边缘
  DType edge_x = center_x + dir_x * scale;
  DType edge_y = center_y + dir_y * scale;

  // 限制在边界内
  edge_x = std::max(0.0, std::min(static_cast<DType>(rows - 1), edge_x));
  edge_y = std::max(0.0, std::min(static_cast<DType>(cols - 1), edge_y));

  return Eigen::Vector2i(static_cast<int>(edge_x), static_cast<int>(edge_y));
}


DType ComputeOptimalV1(DType p0, DType v0, DType a0, DType p1, DType p2,
                        DType v2, DType a2, DType T1, DType T2) {
  // M 项
  DType M = 4.0 * (std::pow(T2, 3) - std::pow(T1, 3)) /
             (T1 * T2 * (std::pow(T1, 2) + std::pow(T2, 2)));

  // N 项
  DType term1 = T2 * T2 * T2 * T2 *
                 (120.0 * (p1 - p0) - 48.0 * v0 * T1 - 6.0 * a0 * T1 * T1);
  DType term2 = T1 * T1 * T1 * T1 *
                 (120.0 * (p2 - p1) - 48.0 * v2 * T2 + 6.0 * a2 * T2 * T2);
  DType denominator_N =
      -18.0 * T1 * T1 * T2 * T2 * (std::pow(T1, 2) + std::pow(T2, 2));
  DType N = (term1 - term2) / denominator_N;

  // P 项
  DType P_numerator = 3.0 * T1 *
                       (std::pow(T2, 5) - 5.0 * std::pow(T1, 5) +
                        4.0 * std::pow(T1, 3) * std::pow(T2, 2));
  DType P_denominator = 16.0 * T2 * (std::pow(T1, 4) + std::pow(T2, 4));
  DType P = P_numerator / P_denominator;

  // Q 项
  DType Q_term1 = (30.0 * (p2 - p1) - 14.0 * v2 * T2 + 2.0 * a2 * T2 * T2) *
                   std::pow(T1, 5);
  DType Q_term2 = (30.0 * (p1 - p0) - 14.0 * v0 * T1 - 2.0 * a0 * T1 * T1) *
                   std::pow(T2, 5);
  DType Q_denominator = 16.0 * T1 * T2 * (std::pow(T1, 4) + std::pow(T2, 4));
  DType Q = (Q_term1 + Q_term2) / Q_denominator;

  // 最终解析表达式
  DType v1 = (P * N + Q) / (1.0 - P * M);
  return v1;
}


std::pair<DType, DType> CalAB(DType kalpha, DType kbeta, DType kgamma,
                                DType balpha, DType bbeta, DType bgamma,
                                DType T) {
  // 计算项 a
  DType a = std::pow(kgamma, 2) + T * kbeta * kgamma +
             (1.0 / 3.0) * std::pow(T, 2) * kalpha * kgamma +
             (1.0 / 4.0) * std::pow(T, 3) * kalpha * kbeta +
             (1.0 / 20.0) * std::pow(T, 4) * std::pow(kalpha, 2) +
             (1.0 / 3.0) * std::pow(T, 2) * std::pow(kbeta, 2);
  // 计算项 b
  DType b = 2.0 * kgamma * bgamma + T * kbeta * bgamma + T * kgamma * bbeta +
             (1.0 / 3.0) * std::pow(T, 2) * 2.0 * kbeta * bbeta +
             (1.0 / 3.0) * std::pow(T, 2) * kalpha * bgamma +
             (1.0 / 3.0) * std::pow(T, 2) * kgamma * balpha +
             (1.0 / 4.0) * std::pow(T, 3) * kalpha * bbeta +
             (1.0 / 4.0) * std::pow(T, 3) * kbeta * balpha +
             (1.0 / 20.0) * std::pow(T, 4) * 2.0 * kalpha * balpha;

  return std::make_pair(a, b);
}

Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> ConvertToDepthImage(const Image& msg) {
  // // 原始深度图大小检查
  // size_t expected_bytes = msg.width * msg.height * 2;
  // if (msg.data.size() != expected_bytes) {
  //   throw std::runtime_error("Invalid input data size.");
  // }

  // 读取为uint16矩阵
  uint32_t elements_per_row = msg.step / 2;
  Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> depth_flat(
    msg.height, elements_per_row);

  for (int i = 0; i < msg.height; ++i) {
    for (int j = 0; j < elements_per_row; ++j) {
      size_t idx = i * msg.step + j * 2;
      uint16_t value;
      if (msg.is_bigendian) {
        value = (static_cast<uint16_t>(msg.data[idx]) << 8) | msg.data[idx + 1];
      } else {
        value = (static_cast<uint16_t>(msg.data[idx + 1]) << 8) | msg.data[idx];
      }
      depth_flat(i, j) = value;
    }
  }

  // 限制深度最大值
  uint16_t max_depth_mm = 10000;
  for (int i = 0; i < depth_flat.rows(); ++i)
    for (int j = 0; j < depth_flat.cols(); ++j)
      depth_flat(i, j) = std::min(depth_flat(i, j), max_depth_mm);

  // 归一化并转换为0~255的uint8图像
  Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> depth_normalized(
      msg.height, msg.width);
  for (int i = 0; i < msg.height; ++i)
    for (int j = 0; j < msg.width; ++j)
      depth_normalized(i, j) = 
          (depth_flat(i, j) / static_cast<DType>(max_depth_mm)) * 255;

  // 下采样（每4个像素采一个，得到32x256）
  int downsample_row = 32;
  int downsample_col = 256;
  Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> downsampled(
      downsample_row, downsample_col);

  for (int i = 0; i < downsample_row; ++i)
    for (int j = 0; j < downsample_col; ++j)
      downsampled(i, j) = depth_normalized(i * 4, j * 4);

  // 截取列 64~191（总共128列）
  Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic> final_img =
      downsampled.block(0, 64, downsample_row, 128);

  // 将值为0的像素替换为最大值（避免后续处理错误）
  DType max_val = final_img.maxCoeff();
  for (int i = 0; i < final_img.rows(); ++i) {
    for (int j = 0; j < final_img.cols(); ++j) {
      if (final_img(i, j) == 0) {
        final_img(i, j) = max_val;
      }
    }
  }
  return final_img;  // 32x128
}

complex_matrix my_ifft2(const complex_matrix &freq) {

    int rows = freq.rows();
    int cols = freq.cols();

    complex_matrix out(rows, cols);

    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // 填充输入
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = freq(i, j).real();
            in[idx][1] = freq(i, j).imag();
        }

    // 2D IFFT
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, fft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // 缩放
    double scale = 1.0 / (rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            out(i, j) = std::complex<double>(fft_out[idx][0] * scale, fft_out[idx][1] * scale);
        }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(fft_out);

    return out;
}

complex_matrix ifft2_exact(const complex_matrix &freq) {
    int rows = freq.rows();
    int cols = freq.cols();
    if (rows == 0 || cols == 0) {
        return complex_matrix();
    }

    complex_matrix out(rows, cols);

    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // 填充输入数据 (row-major, i*cols + j)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = freq(i, j).real();
            in[idx][1] = freq(i, j).imag();
        }

    // 创建 2D IFFT 计划
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, fft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // 缩放因子 1/(rows*cols)，与 numpy ifft2 保持一致
    double scale = 1.0 / (rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            out(i, j) = std::complex<double>(fft_out[idx][0] * scale, fft_out[idx][1] * scale);
        }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(fft_out);

    std::cout << "[ifft2_exact] Filling input array (in):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            in[idx][0] = freq(i, j).real();
            in[idx][1] = freq(i, j).imag();

            std::cout << "  idx=" << idx
                      << " (i=" << i << ", j=" << j << ") "
                      << "in=" << std::fixed << std::setprecision(6)
                      << in[idx][0] << " + " << in[idx][1] << "i"
                      << std::endl;
        }
    }

    return out;
}

complex_matrix ifft2_stepwise(const complex_matrix &freq) {
    int rows = freq.rows();
    int cols = freq.cols();

    if (rows == 0 || cols == 0) {
        return complex_matrix();
    }

    // 中间结果（逐列 IFFT）
    complex_matrix tmp(rows, cols);

    // 1. 逐列 IFFT
    fftw_complex *in_col = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows);
    fftw_complex *out_col = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows);
    fftw_plan plan_col = fftw_plan_dft_1d(rows, in_col, out_col, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            in_col[i][0] = freq(i, j).real();
            in_col[i][1] = freq(i, j).imag();
        }

        fftw_execute(plan_col);

        for (int i = 0; i < rows; ++i) {
            tmp(i, j) = std::complex<double>(out_col[i][0] / rows, out_col[i][1] / rows);
        }
    }

    fftw_destroy_plan(plan_col);
    fftw_free(in_col);
    fftw_free(out_col);

    // 2. 逐行 IFFT
    complex_matrix out(rows, cols);

    fftw_complex *in_row = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * cols);
    fftw_complex *out_row = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * cols);
    fftw_plan plan_row = fftw_plan_dft_1d(cols, in_row, out_row, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            in_row[j][0] = tmp(i, j).real();
            in_row[j][1] = tmp(i, j).imag();
        }

        fftw_execute(plan_row);

        for (int j = 0; j < cols; ++j) {
            out(i, j) = std::complex<double>(out_row[j][0] / cols, out_row[j][1] / cols);
        }
    }

    fftw_destroy_plan(plan_row);
    fftw_free(in_row);
    fftw_free(out_row);

    // Debug: 打印前几行结果
    // std::cout << "[ifft2_stepwise] result:" << std::endl;
    // for (int i = 0; i < std::min(rows, 4); ++i) {
    //     for (int j = 0; j < std::min(cols, 4); ++j) {
    //         std::cout << std::fixed << std::setprecision(6)
    //                   << out(i, j).real() << "+" << out(i, j).imag() << "j  ";
    //     }
    //     std::cout << std::endl;
    // }

    return out;
}

// freq: M x N 的复数矩阵
complex_matrix ifft2_direct(const complex_matrix& freq) {
    int M = freq.rows();
    int N = freq.cols();
    // std::cout << " M N" << M << " " << N << std::endl;
    // 构造 Wm 和 Wn
    complex_matrix Wm(M, M);
    complex_matrix Wn(N, N);

    const std::complex<double> I(0, 1); // 虚数单位 i

    // Wm[k,n] = exp(2j*pi*k*n/M)
    for (int k = 0; k < M; ++k) {
        for (int n = 0; n < M; ++n) {
            Wm(k, n) = exp(2.0 * M_PI * I * double(k * n) / double(M));
        }
    }
    // std::cout << "test wm " << std::endl;
    // std::cout << Wm << std::endl;
    // Wn[l,m] = exp(2j*pi*l*m/N)
    for (int l = 0; l < N; ++l) {
        for (int m = 0; m < N; ++m) {
            Wn(l, m) = exp(2.0 * M_PI * I * double(l * m) / double(N));

        }
    }
    // std::cout << "test Wn" << std::endl;
    // std::cout << Wn << std::endl;
    // X = (1/(M*N)) * Wm^T * freq * Wn^T
    complex_matrix wmt =Wm.transpose();
    complex_matrix wnt = Wn.transpose();

    complex_matrix X = wmt * freq;
    complex_matrix Y = X * wnt;
    // std::cout << "test res " << std::endl;
    // std::cout << "wm transpose " << std::endl;
    // std::cout << wmt << std::endl;
    // std::cout << "wn transpose " << std::endl;
    // std::cout << wnt << std::endl; 
    // std::cout << "res freq " << std::endl;
    // std::cout << freq << std::endl;
    // std::cout << "final res" << std::endl;
    // std::cout << X << std::endl;
    return Y / (M * N);
}
} 