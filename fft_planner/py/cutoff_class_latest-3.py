import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio
from collections import deque
import time
import threading
from path import plot_path_2D_k
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve
import traceback

import rosbag
import rospy

import cv2
np.set_printoptions(threshold=np.inf)   
np.set_printoptions(
    threshold=np.inf, # 不省略，全部打印
    precision=6,      # 小数点后6位
    suppress=True,    # 不使用科学计数法
    linewidth=20000,    # 每行最大字符数，不自动换行
    formatter={'complex_kind': lambda x: f'{x.real:.6f}+{x.imag:.6f}j'}
)
def map_to_edge(matrix_shape, index):
    """
    将角标映射到矩阵边缘
    :param matrix_shape: 矩阵的形状 (rows, cols)
    :param index: 输入角标 (x, y)
    :return: 映射后的角标 (x, y)
    """
    rows, cols = matrix_shape
    x, y = index

    # 计算矩阵中心
    center_x, center_y = rows / 2, cols / 2

    # 如果角标在矩阵范围内，直接返回
    if 0 <= x < rows and 0 <= y < cols:
        return int(x), int(y)

    # 计算方向向量
    direction_x = x - center_x
    direction_y = y - center_y

    # 计算比例因子，找到与矩阵边界的交点
    if direction_x != 0:
        scale_x = (0 - center_x) / direction_x if direction_x < 0 else (rows - 1 - center_x) / direction_x
    else:
        scale_x = float('inf')  # 垂直方向

    if direction_y != 0:
        scale_y = (0 - center_y) / direction_y if direction_y < 0 else (cols - 1 - center_y) / direction_y
    else:
        scale_y = float('inf')  # 水平方向

    # 选择最小的比例因子，确保交点在矩阵边界
    scale = min(scale_x, scale_y)

    # 计算映射后的角标
    edge_x = center_x + direction_x * scale
    edge_y = center_y + direction_y * scale

    # 将结果限制在矩阵边界范围内
    edge_x = max(0, min(rows - 1, edge_x))
    edge_y = max(0, min(cols - 1, edge_y))

    return np.array([edge_x, edge_y], dtype=int)


def compute_optimal_v1(p0, v0, a0, p1, p2, v2, a2, T1, T2):
    """
    计算最优中间速度v₁，苏黎世法
    参数:
        p0, v0, a0 : 初始状态
        p1         : 中间点位置
        p2, v2, a2 : 终末状态
        T1, T2     : 两段轨迹时间

    返回:
        v1 : 解析解
    """
    M = 4 * (T2**3 - T1**3) / (T1*T2 * (T1**2 + T2**2))
    N = (T2**4 * (120 * (p1 - p0) - 48*v0*T1 - 6*a0*T1**2) - T1**4 * (120*(p2 - p1) - 48*v2*T2 + 6*a2*T2**2)) / (-18*T1*T1*T2*T2 * (T1**2 + T2**2))
    P = 3 * T1 * (T2**5 - 5 * T1**5 + 4 * T1**3 * T2**2) / (16 * T2 * (T1**4 + T2**4))
    Q = ((30 * (p2 - p1) - 14 * v2 * T2 + 2 * a2 * T2**2) * T1**5 + (30 * (p1 - p0) - 14 * v0 * T1 - 2 * a0 * T1**2) * T2**5) / (16 * T1 * T2 * (T1**4 + T2**4))

    return (P*N + Q) / (1 - P*M)


# def compute_optimal_v1(p0, v0, a0, p1, p2, v2, a2, T1, T2):
#     """
#     计算最优中间速度v₁，GPT五次多项式函数
#     参数:
#         p0, v0, a0 : 初始状态
#         p1         : 中间点位置
#         p2, v2, a2 : 终末状态
#         T1, T2     : 两段轨迹时间
#
#     返回:
#         v1 : 解析解
#     """
#     M = 4 * (T2 - T1) / (T1 * T2)
#     N = (T2**3 * (-10 * (p1 - p0) + 4 * v0 * T1 + 0.5 * a0 * T1**2) - T1**3 * (-10 * (p2 - p1) + 4 * v2 * T2 - 0.5 * a2 * T2**2)) / (1.5 * T1**2 * T2**2 * (T1 + T2))
#
#     P = 1.5 * T1 * T2 * (T1**2 + T2**2) / (8 * (T1**3 + T2**3))
#     Q = (T1**4 * (15 * (p2 - p1) - 7 * v2 * T2 + a2 * T2**2) + T2**4 * (15 * (p1 - p0) - 7 * v0 * T1 - a0 * T1**2)) / (8 * T1 * T2 * (T1**3 + T2**3))
#
#     return (P*N + Q) / (1 - P*M)


# def compute_optimal_v1(p0, v0, a0, p1, p2, v2, a2, T1, T2):
#     """
#     计算最优中间速度v₁
#     三次多项式的情况
#     参数:
#         p0, v0, a0 : 初始状态
#         p1         : 中间点位置
#         p2, v2, a2 : 终末状态
#         T1, T2     : 两段轨迹时间
#
#     返回:
#         v1 : 解析解
#     """
#     # 计算位移向量
#     delta_p1 = p1 - p0
#     delta_p2 = p2 - p1
#
#     # 计算公共分母
#     denominator = T1 ** 3 + T2 ** 3 + 3 * T1 * T2 ** 2 + 3 * T1 ** 2 * T2
#
#     # 向量化计算分子项
#     term1 = T2 ** 3 * (3 * delta_p1 - T1 * v0)
#     term2 = T1 ** 3 * (3 * delta_p2 + T2 * v2)
#     term3 = T1 * T2 ** 2 * (a0 * T1)
#     term4 = T1 ** 2 * T2 * (a2 * T2)
#
#     # 合并分子并计算最终速度
#     v1 = (term1 + term2 + term3 + term4) / denominator
#
#     return v1

def cal_ab(kalpha, kbeta, kgamma, balpha, bbeta, bgamma, T):
    a = (kgamma**2 + T*kbeta*kgamma + 1/3*(T**2)*kalpha*kgamma + 1/4*(T**3)*kalpha*kbeta + 1/20*(T**4)*(kalpha**2) +
         1/3*(T**2)*kbeta**2)  # 之前发现少写了一项
    b = 2*kgamma*bgamma + T*kbeta*bgamma + T*kgamma*bbeta + 1/3*(T**2)*2*kbeta*bbeta + 1/3*(T**2)*kalpha*bgamma + 1/3*(T**2)*kgamma*balpha + \
        1/4*(T**3)*kalpha*bbeta + 1/4*(T**3)*kbeta*balpha + 1/20*(T**4)*2*kalpha*balpha
    return a, b

class optim_solver:
    def __init__(self, p0, a, b, r, safe, rotation, angle_ptx, gmax, gmin, shape, centre):
        self.shape = shape
        self.centre = centre

        self.p0 = p0
        self.a = a
        self.b = b
        self.r = r

        self.r_g = self.shape[1] / 2 / 4 / 2
        self.gmin_range = np.array([self.r_g/2, self.shape[0] - self.r_g/2], dtype=int)

        self.safe = safe
        self.gradient = np.zeros([self.shape[1], self.shape[1]])  # 滤波后的安全域梯度
        self.rotation = rotation
        self.angle_ptx = angle_ptx
        self.gmax = gmax
        self.gmin = gmin

        self.refox = 0
        self.refoy = 0

    def cal_gmax_gmin(self):
        self.gmax = np.max(self.gradient)
        self.gmin = np.min(self.gradient)
        # self.gmin = np.min(self.gradient[self.gmin_range[0]:self.gmin_range[1],:])


    def penalty_debug(self):
        '''计算安全域惩罚项的变化'''
        '''局部坐标系'''
        pmat = np.zeros([self.shape[0], self.shape[1]], dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # 计算本地坐标系下的下一个目标点
                angle_horizontal = self.angle_ptx * (self.centre[1] - j)  # 水平角，左为正方向
                angle_vertical = np.pi / 2 - self.angle_ptx * (self.centre[0] - i)

                xx = 5.65 * np.sin(angle_vertical) * np.cos(angle_horizontal)
                yy = 5.65 * np.sin(angle_vertical) * np.sin(angle_horizontal)
                zz = 5.65 * np.cos(angle_vertical)

                [x, y, z] = self.rotation.apply(np.array([xx, yy, zz]))  # 本地到世界坐标系
                pmat[i][j] = self.penalty(x, y, z)
                # pmat[i][j] = min(pmat[i][j], 2+self.r**2)  # 去除过大的边界值

        plt.figure(figsize=(8, 2))
        plt.imshow(pmat)
        plt.title("penalty")
        plt.colorbar()
        plt.show()

    def penalty_debug_mx(self):
        '''计算安全域惩罚项的变化'''
        '''局部坐标系'''
        pmat = np.zeros([self.shape[0], self.shape[1]], dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                pmat[i][j] = self.penalty_safe_debug(i, j)
                # pmat[i][j] = min(pmat[i][j], 2+self.r**2)  # 去除过大的边界值

        plt.figure(figsize=(8, 2))
        plt.imshow(pmat)
        plt.title("penalty")
        plt.colorbar()
        plt.show()

    def dynamics_debug(self):
        '''动力学成本比较大，但惩罚项函数的尺度需要和圆周约束项匹配'''
        '''局部坐标系'''
        pmat = np.zeros([self.shape[0], self.shape[1]], dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # 计算本地坐标系下的下一个目标点
                angle_horizontal = self.angle_ptx * (self.centre[1] - j)  # 水平角，左为正方向
                # 球坐标与z轴的夹角，XOY平面为90°，当矩阵角标小于240时，更接近z轴正方向
                angle_vertical = np.pi / 2 - self.angle_ptx * (self.centre[0] - i)

                xx = 5.65 * np.sin(angle_vertical) * np.cos(angle_horizontal)
                yy = 5.65 * np.sin(angle_vertical) * np.sin(angle_horizontal)
                zz = 5.65 * np.cos(angle_vertical)

                [x, y, z] = self.rotation.apply(np.array([xx, yy, zz]))  # 本地到世界坐标系
                pmat[i][j] = self.a[0] * x**2 + self.b[0]*x + self.a[1] * y**2 + self.b[1]*y + self.a[2] * z**2 + self.b[2]*z
                # pmat[i][j] = min(pmat[i][j], 2+self.r**2)  # 去除过大的边界值

        plt.figure(figsize=(8, 2))
        plt.imshow(pmat)
        plt.title("dynamics")
        plt.colorbar()
        plt.show()

    def dynamics_sample(self):
        '''采样计算动力学成本的最大值和最小值，返回pmax, pmin'''
        '''局部坐标系'''
        pmax = -np.inf
        pmin = np.inf
        step = int(self.shape[0]/2)
        for i in [0, int(self.shape[0]/2), self.shape[0]-1]:
            for j in range(0, self.shape[1], step):
                # 计算本地坐标系下的下一个目标点
                angle_horizontal = self.angle_ptx * (self.centre[1] - j)  # 水平角，左为正方向
                # 球坐标与z轴的夹角，XOY平面为90°，当矩阵角标小于240时，更接近z轴正方向
                angle_vertical = np.pi / 2 - self.angle_ptx * (self.centre[0] - i)

                xx = 5.65 * np.sin(angle_vertical) * np.cos(angle_horizontal)
                yy = 5.65 * np.sin(angle_vertical) * np.sin(angle_horizontal)
                zz = 5.65 * np.cos(angle_vertical)

                [x, y, z] = self.rotation.apply(np.array([xx, yy, zz]))  # 本地到世界坐标系
                cost = self.a[0] * x**2 + self.b[0]*x + self.a[1] * y**2 + self.b[1]*y + self.a[2] * z**2 + self.b[2]*z
                pmax = max(cost, pmax)
                pmin = min(cost, pmin)

        return pmax, pmin


    def penalty(self, x, y, z, d_x=0, d_y=0):
        # return 2
        '''惩罚项，要求轨迹在安全域内'''
        '''d_x, d_y 用于计算偏导数使用'''
        # 将 (x, y, z) 映射到离散网格点
        # p_ref_local = self.rotation.inv().apply(p_ref - p0)

        local = self.rotation.inv().apply(np.array([x, y, z]))  # !====重要，需要减去初始位置p0
        e_local = local / np.linalg.norm(local)

        angle_horizontal1 = np.arctan(e_local[1] / e_local[0])  # phi
        ref_y = d_y + self.centre[1] - angle_horizontal1 / self.angle_ptx

        angle_vertical1 = np.arccos(e_local[2])  # theta
        ref_x = d_x + self.centre[0] + (angle_vertical1 - np.pi / 2) / self.angle_ptx

        if ref_x <= 1 or ref_x >= self.shape[0] -2 or ref_y <= 1 or ref_y >= self.shape[1] -2:
            # return self.k * 2*self.shape[0]*10000 + 2  # shape[1]更激进
            return self.percent * 2 + 2  # shape[1]更激进

        if self.safe[round(ref_x), round(ref_y)] > 0.99:
            return 2

        # 梯度与球面约束的梯度匹配
        '''此处虽然做了归一化，但系数仍然是人手工赋值'''
        '''通过测试发现，每次迭代后，动力学成本有所下降，但安全域成本不变'''
        '''最后会导致梯度不匹配，反而造成震荡'''
        '''
        解决方法：
        （1）[尚可]工程上估计运动学成本的最大值和最小值（比如按照非常粗糙的分辨率采样）
        （2）[效果不好]每次迭代时，对安全域成本乘以衰减倍率，如0.5。因为约往后迭代，靶点落在安全域的概率越高，应该不太需要增加额外的安全约束。并且能够平衡动力学成本的下降
        '''
        # return 2 + self.k * self.shape[0]*10000 * (1 - (self.gradient[round(ref_x), round(ref_y)] - self.gmin) / (self.gmax - self.gmin))  # shape[1]更激进
        return 2 + self.percent * (1 - (self.gradient[round(ref_x), round(ref_y)] - self.gmin) / (self.gmax - self.gmin))  # shape[1]更激进


    def penalty_safe(self, x, y, z, d_x=0, d_y=0):
        # return 2
        '''惩罚项，要求轨迹在安全域内'''
        local = self.rotation.inv().apply(np.array([x, y, z]))  # !====重要，需要减去初始位置p0
        e_local = local / np.linalg.norm(local)

        angle_horizontal1 = np.arctan(e_local[1] / e_local[0])  # phi
        ref_y = d_y + self.centre[1] - angle_horizontal1 / self.angle_ptx

        angle_vertical1 = np.arccos(e_local[2])  # theta
        ref_x = d_x + self.centre[0] + (angle_vertical1 - np.pi / 2) / self.angle_ptx

        if ref_x < 0 or ref_x > self.shape[0] - 1 or ref_y < 0 or ref_y > self.shape[1] - 1:
            return 2 * self.shape[0] * 10000 + 2  # shape[1]更激进

            ref = map_to_edge(self.shape, (ref_x, ref_y))
            k = (self.shape[0] * 10000 + 2) / np.linalg.norm(self.centre - ref)
            return k * np.linalg.norm(self.centre - np.array([ref_x, ref_y])) + 2

            # return 2*self.shape[0] * 10000 + 2  # shape[1]更激进

        # try:
        #     if self.safe[round(ref_x), round(ref_y)] > 0.99:
        #         return 2
        # except Exception as e:
        #     print(e)
        #     print("ref_x:", ref_x)
        #     print("ref_y:", ref_y)
        #     print("e_local:", e_local)
        #     print("angle_horizontal1:", angle_horizontal1)
        #     print("angle_vertical1:", angle_vertical1)

        try:
            if self.safe[round(ref_x), round(ref_y)] > 0.99:
                return 2
        except Exception as e:
            traceback.print_exc()  # 打印完整的错误堆栈信息
            print("Exception occurred:", e)
            print("x:", x)
            print("y:", y)
            print("z:", z)
            print("ref_x:", ref_x)
            print("ref_y:", ref_y)
            print("e_local:", e_local)
            print("angle_horizontal1:", angle_horizontal1)
            print("angle_vertical1:", angle_vertical1)
            exit(0)


        return 2 + self.shape[0] * 10000 * (1 - (self.gradient[round(ref_x), round(ref_y)] - self.gmin) / (self.gmax - self.gmin))  # shape[1]更激进


    def penalty_safe_debug(self, mx, my, d_x=0, d_y=0):
        # return 2
        mx += d_x
        my += d_y

        '''惩罚项，要求轨迹在安全域内'''
        if mx < 0 or mx > self.shape[0] - 1 or my < 0 or my > self.shape[1] - 1:
            return 1 + 2  # shape[1]更激进

        if self.safe[round(mx), round(my)]:
            return 0

        return (1 - (self.gradient[mx, my] - self.gmin) / (self.gmax - self.gmin))  # shape[1]更激进

    def penalty_safe_mx(self, mx, my, d_x=0, d_y=0):
        # return 2
        mx += d_x
        my += d_y

        '''惩罚项，要求轨迹在安全域内'''
        if mx < 0 or mx > self.shape[0] - 1 or my < 0 or my > self.shape[1] - 1:
            return 1 * self.shape[0] * 10000 + 2  # shape[1]更激进

            ref = map_to_edge(self.shape, (ref_x, ref_y))
            k = (self.shape[0] * 10000 + 2) / np.linalg.norm(self.centre - ref)
            return k * np.linalg.norm(self.centre - np.array([ref_x, ref_y])) + 2

            # return 2*self.shape[0] * 10000 + 2  # shape[1]更激进

        if self.safe[round(mx), round(my)]:
            return 2

        return 2 + self.shape[0] * 10000 * (1 - (self.gradient[mx, my] - self.gmin) / (self.gmax - self.gmin))  # shape[1]更激进

    # def numerical_derivative_mx(self, f, mx, my, axis):
    #     '''
    #     计算penalty函数在指定轴上的数值偏导数
    #
    #     f: 惩罚函数
    #     x, y, z: 当前点,世界坐标系
    #     axis: 需要计算偏导数的轴, 'mx', 'my'
    #     epsilon: 用于有限差分法的增量
    #     '''
    #     if axis == 'mx':
    #         k1 = abs(int(mx) + 1 - mx)
    #         k2 = abs(int(mx) - mx)
    #         return k1 * (f(int(mx)+1, round(my), d_x=1, d_y=0) - f(int(mx)+1, round(my), d_x=-1, d_y=0)) / (2 * self.angle_ptx) + \
    #                k2 * (f(int(mx), round(my), d_x=1, d_y=0) - f(int(mx), round(my), d_x=-1, d_y=0)) / (2 * self.angle_ptx)
    #     elif axis == 'my':
    #         k1 = abs(int(my) + 1 - my)
    #         k2 = abs(int(my) - my)
    #         return k1 * (f(round(mx), int(my)+1, d_x=0, d_y=1) - f(round(mx), int(my)+1, d_x=0, d_y=-1)) / (2 * self.angle_ptx) + \
    #                k2 * (f(round(mx), int(my), d_x=0, d_y=1) - f(round(mx), int(my), d_x=0, d_y=-1)) / (2 * self.angle_ptx)
    #     else:
    #         raise ValueError("Invalid axis. Choose 'theta', or 'phi'.")

    def numerical_derivative_mx(self, f, mx, my, axis):
        '''
        计算penalty函数在指定轴上的数值偏导数

        f: 惩罚函数
        x, y, z: 当前点,世界坐标系
        axis: 需要计算偏导数的轴, 'mx', 'my'
        epsilon: 用于有限差分法的增量
        '''
        if axis == 'mx':
            return (f(mx, my, d_x=1, d_y=0) - f(mx, my, d_x=-1, d_y=0)) / (2)
        elif axis == 'my':
            return (f(mx, my, d_x=0, d_y=1) - f(mx, my, d_x=0, d_y=-1)) / (2)
        else:
            raise ValueError("Invalid axis. Choose 'theta', or 'phi'.")


    def numerical_derivative_mx_single(self, f, mx, my, axis):
        '''
        计算penalty函数在指定轴上的数值偏导数

        f: 惩罚函数
        x, y, z: 当前点,世界坐标系
        axis: 需要计算偏导数的轴, 'mx', 'my'
        epsilon: 用于有限差分法的增量
        '''
        if axis == 'mx':
            return (f(mx, my, d_x=1, d_y=0) - f(mx, my, d_x=0, d_y=0))
        elif axis == 'my':
            return (f(mx, my, d_x=0, d_y=1) - f(mx, my, d_x=0, d_y=0))
        else:
            raise ValueError("Invalid axis. Choose 'theta', or 'phi'.")

    def numerical_derivative(self, f, x, y, z, axis):
        '''
        计算penalty函数在指定轴上的数值偏导数

        f: 惩罚函数
        x, y, z: 当前点,世界坐标系
        axis: 需要计算偏导数的轴, 'theta', 'phi'
        epsilon: 用于有限差分法的增量
        '''
        if axis == 'theta':
            return (f(x, y, z, d_x=1, d_y=0) - f(x, y, z, d_x=-1, d_y=0)) / (2 * self.angle_ptx)
        elif axis == 'phi':
            return -(f(x, y, z, d_x=0, d_y=1) - f(x, y, z, d_x=0, d_y=-1)) / (2 * self.angle_ptx)
        else:
            raise ValueError("Invalid axis. Choose 'theta', or 'phi'.")

    def numerical_derivative_single(self, f, x, y, z, axis):
        '''
        计算penalty函数在指定轴上的数值偏导数
        仅计算迭代梯度方向的偏导数，而不是连续函数的导数，这样更符合梯度下降的过程

        f: 惩罚函数
        x, y, z: 当前点,世界坐标系
        axis: 需要计算偏导数的轴, 'theta', 'phi'
        epsilon: 用于有限差分法的增量
        '''
        if axis == 'theta':
            return (f(x, y, z, d_x=1, d_y=0) - f(x, y, z, d_x=0, d_y=0)) / (self.angle_ptx)
        elif axis == 'phi':
            return -(f(x, y, z, d_x=0, d_y=1) - f(x, y, z, d_x=0, d_y=0)) / (self.angle_ptx)
        else:
            raise ValueError("Invalid axis. Choose 'theta', or 'phi'.")

    def equations(self, vars):
        '''拉格朗日法求解成本函数极值，全局坐标系'''
        # vars = [x, y, z, lambda]
        theta, phi, lam = vars
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(theta)

        # 加入惩罚项
        penalty_val = self.penalty(x, y, z)

        # 计算惩罚项的偏导数
        d_penalty_dtheta = self.numerical_derivative(self.penalty, x, y, z, 'theta')
        d_penalty_dphi = self.numerical_derivative(self.penalty, x, y, z, 'phi')

        # 约束条件
        eq3 = penalty_val - 2

        # 拉格朗日函数
        eq1 = (2 * self.a[0] * x + self.b[0]) * (self.r * np.cos(theta) * np.cos(phi)) + \
              (2 * self.a[1] * y + self.b[1]) * (self.r * np.cos(theta) * np.sin(phi)) + \
              (2 * self.a[2] * z + self.b[2]) * (-self.r * np.sin(phi)) + \
              + lam * d_penalty_dtheta  # dL/dtheta = 0

        eq2 = (2 * self.a[0] * x + self.b[0]) * (-self.r * np.sin(theta) * np.sin(phi)) + \
              (2 * self.a[1] * y + self.b[1]) * (self.r * np.sin(theta) * np.cos(phi)) + \
              + lam * d_penalty_dphi  # dL/dphi = 0

        return [eq1, eq2, eq3]


    def equations_dynamics(self, vars):
        '''拉格朗日法求解成本函数极值，全局坐标系'''
        # vars = [x, y, z, lambda]
        theta, phi = vars
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(theta)

        # 拉格朗日函数
        eq1 = (2 * self.a[0] * x + self.b[0]) * (self.r * np.cos(theta) * np.cos(phi)) + \
              (2 * self.a[1] * y + self.b[1]) * (self.r * np.cos(theta) * np.sin(phi)) + \
              (2 * self.a[2] * z + self.b[2]) * (-self.r * np.sin(phi))
              # dL/dtheta = 0

        eq2 = (2 * self.a[0] * x + self.b[0]) * (-self.r * np.sin(theta) * np.sin(phi)) + \
              (2 * self.a[1] * y + self.b[1]) * (self.r * np.sin(theta) * np.cos(phi))
              # dL/dphi = 0

        return [eq1, eq2]

    # def equations_safe(self, vars):
    #     '''
    #     拉格朗日法求解成本函数极值，全局坐标系'''
    #     # vars = [x, y, z, lambda]
    #     theta, phi, lam = vars
    #     x = self.r * np.sin(theta) * np.cos(phi)
    #     y = self.r * np.sin(theta) * np.sin(phi)
    #     z = self.r * np.cos(theta)
    #
    #     # 加入惩罚项
    #     penalty_val = self.penalty_safe(x, y, z)
    #
    #     # 计算惩罚项的偏导数
    #     d_penalty_dtheta = self.numerical_derivative(self.penalty_safe, x, y, z, 'theta')
    #     d_penalty_dphi = self.numerical_derivative(self.penalty_safe, x, y, z, 'phi')
    #
    #     # 约束条件
    #     eq3 = penalty_val - 2
    #
    #     # 拉格朗日函数
    #     eq1 = (theta - self.theta0) + lam * d_penalty_dtheta  # dL/dtheta = 0
    #     eq2 = (phi - self.phi0) + lam * d_penalty_dphi  # dL/dphi = 0
    #
    #     return [eq1, eq2, eq3]

    def equations_safe_mx(self, vars):
        '''
        拉格朗日法求解成本函数极值，全局坐标系'''
        # vars = [x, y, z, lambda]
        refox, refoy, lam = vars

        refox = round(refox)
        refoy = round(refoy)

        # 加入惩罚项
        penalty_val = self.penalty_safe_mx(refox, refoy)

        # 计算惩罚项的偏导数 沿mx，my增加的方向 单方向的导数太激进，连续的导数太保守
        d_penalty_dmx = self.numerical_derivative_mx(self.penalty_safe_mx, refox, refoy, 'mx')
        d_penalty_dmy = self.numerical_derivative_mx(self.penalty_safe_mx, refox, refoy, 'my')

        # 约束条件
        eq3 = penalty_val - 2

        # 拉格朗日函数
        eq1 = (refox - self.refox) + lam * d_penalty_dmx  # dL/dtheta = 0
        eq2 = (refoy - self.refoy) + lam * d_penalty_dmy  # dL/dphi = 0

        return [eq1, eq2, eq3]

    # def solve(self, initial_p):
    #     '''经典拉格朗日法'''
    #     # 可能需要手动设计雅可比矩阵，不然在靠近边界时会导致惩罚项太大而梯度过大
    #     sol = fsolve(self.equations, initial_p, epsfcn=1e-4, xtol=1e-2, col_deriv=True)  # 可用：epsfcn=1e-2, xtol=1e-3
    #     # epsfcn=1e-1, xtol=1e-2
    #     theta_sol, phi_sol, _ = sol
    #     return theta_sol, phi_sol

    # def solve(self, initial_p):
    #     '''先计算动力学成本，再计算安全域成本'''
    #     sol = fsolve(self.equations_dynamics, initial_p[0:2], epsfcn=1e-1, xtol=1e-2, col_deriv=True)
    #     # 充足：epsfcn=1e-2, xtol=1e-3
    #     # 可用：epsfcn=1e-1, xtol=1e-2
    #     theta_sol, phi_sol = sol
    #
    #     self.theta0 = theta_sol
    #     self.phi0 = phi_sol
    #
    #     # 【严重】该求解结果随self.angle_ptx极度敏感，过程可能有错
    #     # 原因1：单方向导数
    #     # 原因2：边界值梯度异常
    #     sol = fsolve(self.equations_safe, np.array([theta_sol, phi_sol, 0.0]), epsfcn=1e-4, xtol=1e-2, col_deriv=True)
    #     # 充足：epsfcn=1e-4, xtol=1e-5
    #     # 多数充足：epsfcn=1e-4, xtol=1e-2，求解结果相差不大
    #     # epsfcn=1e-3, xtol=1e-4，降低后部分求解点差异较大
    #     # 关键在epsfcn
    #     theta_sol, phi_sol, _ = sol
    #
    #     return theta_sol, phi_sol


    def solve_safe(self):
        '''使用手动梯度上升来将点迭代至最近的安全域'''
        mx = self.refox
        my = self.refoy
        # threshold = 1 / (np.pi * self.r_g**2)
        if mx < 0 or mx > self.shape[0] - 1 or my < 0 or my > self.shape[1] - 1:
            mx, my = map_to_edge(self.shape, (mx, my))

        for tt in range(0, 100):  # 最大迭代次数
            if self.safe[mx, my]:
                return mx, my

            ex_mx = mx
            ex_my = my

            # 计算当前点的梯度
            # g = np.array([
            #     (self.gradient[mx + 1, my] - self.gradient[mx - 1, my]) / 2,
            #     (self.gradient[mx, my + 1] - self.gradient[mx, my - 1]) / 2
            # ])

            if 0 < mx < self.shape[0] - 1 and 0 < my < self.shape[1] - 1:
                g = np.array([
                    (self.gradient[mx + 1, my] - self.gradient[mx - 1, my]) / 2,
                    (self.gradient[mx, my + 1] - self.gradient[mx, my - 1]) / 2
                ])
            elif mx == 0 and 0 < my < self.shape[1] - 1:
                g = np.array([
                    (self.gradient[mx + 1, my] - self.gradient[mx, my]),
                    (self.gradient[mx, my + 1] - self.gradient[mx, my - 1]) / 2
                ])
            elif mx == self.shape[0] - 1 and 0 < my < self.shape[1] - 1:
                g = np.array([
                    (self.gradient[mx, my] - self.gradient[mx - 1, my]),
                    (self.gradient[mx, my + 1] - self.gradient[mx, my - 1]) / 2
                ])
            elif 0 < mx < self.shape[0] - 1 and my == 0:
                g = np.array([
                    (self.gradient[mx + 1, my] - self.gradient[mx - 1, my]) / 2,
                    (self.gradient[mx, my + 1] - self.gradient[mx, my])
                ])
            elif 0 < mx < self.shape[0] - 1 and my == self.shape[1] - 1:
                g = np.array([
                    (self.gradient[mx + 1, my] - self.gradient[mx - 1, my]) / 2,
                    (self.gradient[mx, my] - self.gradient[mx, my - 1])
                ])

            g_norm = np.linalg.norm(g)

            # if g_norm < threshold:
            #     mx += round(g[0] / g_norm * self.r_g)
            #     my += round(g[1] / g_norm * self.r_g)
            #
            # else:
            #     d = -0.5 * (g_norm / threshold) + self.r_g
            #     # d = np.sqrt(self.r_g**2 - (g_norm / threshold)**2 / 4)
            #     mx += round(g[0] / g_norm * d)
            #     my += round(g[1] / g_norm * d)

            d = self.r_g / (self.gmin - self.gmax) * (self.gradient[mx, my] - self.gmin) + self.r_g
            mx += round(g[0] / g_norm * d)
            my += round(g[1] / g_norm * d)

            if abs(mx - ex_mx) < 1 and abs(my - ex_my) < 1:
                # 如果没有移动，则停止迭代
                return mx, my

            if mx < 0 or mx > self.shape[0] - 1 or my < 0 or my > self.shape[1] - 1:
                mx, my = map_to_edge(self.shape, (mx, my))

        return mx, my

    def solve(self, initial_p):
        '''先计算动力学成本，再计算安全域成本'''
        sol = fsolve(self.equations_dynamics, initial_p[0:2], epsfcn=1e-1, xtol=1e-2, col_deriv=True)
        # 充足：epsfcn=1e-2, xtol=1e-3
        # 可用：epsfcn=1e-1, xtol=1e-2
        theta_sol, phi_sol = sol

        x_sol = self.r * np.sin(theta_sol) * np.cos(phi_sol)
        y_sol = self.r * np.sin(theta_sol) * np.sin(phi_sol)
        z_sol = self.r * np.cos(theta_sol)
        po = np.array([x_sol, y_sol, z_sol], dtype=float)

        po_local = self.rotation.inv().apply(po)  # 本地坐标系
        eo_local = po_local / np.linalg.norm(po_local)  # 最佳靶点的单位向量

        # 计算最佳方向的投影
        angle_horizontal1 = np.arctan(eo_local[1] / eo_local[0])  # phi
        refoy = (self.centre[1] - angle_horizontal1 / self.angle_ptx)  # y

        angle_vertical1 = np.arccos(eo_local[2])  # theta
        refox = (self.centre[0] + (angle_vertical1 - np.pi / 2) / self.angle_ptx)  # x

        self.refox = round(refox)
        self.refoy = round(refoy)

        # self.refox = 6
        # self.refoy = 66


        # 求解器太不稳定，不如自己直接写梯度下降的搜索?
        # 由于滤波器是归一化的，因此可以根据滤波器面积大小，以及滤波后点的数值来判断出梯度方向的大致距离
        # 然后根据xy梯度的比例进行下降搜索
        # sol = fsolve(self.equations_safe_mx, np.array([refox, refoy, 0.0]), epsfcn=1e-4, xtol=1e-3, col_deriv=True)
        #
        # mx, my, _ = sol

        mx, my = self.solve_safe()
        return mx, my



class FFT_plan:
    def __init__(self, mode=0):
        # mode设置：-1 绘制对比图 # 0 静默模式 # 1 文字输出 # 2 plot debug
        # 默认为0
        # 仿真时应当设为0
        # debug时可以设置为1

        # 以下参数需要按照无人机性能调整
        self.max_spd = 20.  # 无人机能达到的最大速度（速率）
        self.max_acc = 15.  # 无人机能达到的最大加速度（加速率）
        self.exp_spd = 4.  # 预期飞行速度
        self.kt = 0.25  # 轨迹点间隔时间/s
        '''
        设计该参数时需要考虑两个因素：
        1. 控制轨迹长度（途径点间隔）：self.kt=途径点间隔/self.max_spd
        例如，途径点间隔=1m，设计时速为10m/s，则self.kt=1/10=0.1s
        2. 控制加加速度的最大突变：self.kt=max_acc / max_jerk
        '''

        # ***以下为【可配置】参数，需要根据表现情况调整
        self.mode = mode  # 0:无输出运行，用于最小延迟上机实验，1:debug，会输出所有打印，2:会输出plt图像
        self.epsilon = 3.71  # 一个小数值，用于平抑深度相机的误差，数值越小越严格，数值越大，安全域的边界越大，数值太大会导致危险
        #  1.9-45*180: 255 / self.dis_max * self.cutoff[j] * (1 / (3.14* 5*5))
        # 3.71-32*128
        self.ang_acc = 1  # 第一个截断面：广度优先搜索的像素精度。数值太小会计算量太大，过大则会遗漏安全方向
        self.dtsep = 2  # 第二个截断面：求取极值的下采样，数值太小会计算量太大，过大则会遗漏安全方向。强烈不建议取1
        self.replan_time = 0.5   # 强制重新规划时间，避免持续跟踪一个远端轨迹造成的问题  需要根据无人机速度设定


        # ***以下参数【不可修改】，否则会出现逻辑问题
        self.FOV = np.array([45., 180.]) * np.pi / 180  # 视场角
        self.shape = np.array([32, 128])
        self.mx_centre = np.array([self.shape[0] / 2, self.shape[1] / 2], dtype=int)

        self.iter = 5  # 迭代次数 5够用
        self.dis_max = 10.  # 深度相机能够输入的最远距离

        self.cutoff = [5.65, 10.]  # 截断距离
        self.r = np.array([7 / 2, 4 / 2], dtype=int)  # 截断距离对应的滤波器半径

        self.angle_ptx = 1.40625 * np.pi / 180  # 角度精度设置为1 deg
        self.path_len = len(self.cutoff)

        # 计算不同截断距离下的三种成本权重
        # 行：2个截断距离
        # 列：空旷成本，参考方向成本
        self.w_costs = np.ones([self.path_len, 4])
        for i in range(self.path_len):
            self.w_costs[i][0] = 1.  # 动力学成本
            self.w_costs[i][1] = 1.  # 空旷成本
            self.w_costs[i][2] = 1.  # 参考方向成本
            self.w_costs[i][3] = 0.5  # 上一个时刻的参考方向成本
            self.w_costs[i] = self.w_costs[i] / np.sum(self.w_costs[i])

        # 加载滤波器
        self.H_S = []
        for i in range(2):
            try:
                mat_file_path = './H{}-f-S.mat'.format(i + 1)
                # mat_file_path = './H{}-f-S-全分辨率.mat'.format(i + 1)
                self.H_S.append(np.array(scio.loadmat(mat_file_path)['HS']))
            except Exception as e:
                print(f"Error loading file: {mat_file_path}")
                print(f"Exception: {e}")

        # 加载滤波器
        self.H_D = []
        for i in range(2):
            try:
                # mat_file_path = './H{}-f-D-全分辨率.mat'.format(i + 1)
                mat_file_path = './H{}-f-D.mat'.format(i + 1)
                self.H_D.append(np.array(scio.loadmat(mat_file_path)['HD']))
            except Exception as e:
                print(f"Error loading file: {mat_file_path}")
                print(f"Exception: {e}")

        self.Hshape = self.H_D[0].shape[0]

        self.safe = np.zeros([self.path_len, self.shape[0], self.shape[1]], dtype=bool)  # 用于存储FFT滤波后的安全域
        self.rotation = None
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.linar_v = np.array([0.0, 0.0, 0.0])
        self.linar_a = np.array([0.0, 0.0, 0.0])  # 加速度，由于odom无输出，因此恒定为0
        self.target_direction = np.array([1.0, 0.0, 0.0])
        self.target_distance = 0.0
        self.target = np.array([10.0, 0., 0.])
        self.target_vector = np.array([10.0, 0., 0.])
        self.approaching = False  # 快要到达终点的标志位

        self.last_end_point = np.array([0.0, 0.0, 0.0])  # 上一个终点位置，平移为上时刻的位置，旋转为全局坐标系
        self.last_end_val = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])  # 上一个位置，IMU读数，首次规划为0

        # 定义优化器
        self.solver = optim_solver(p0=np.zeros(3),
                                  a=np.zeros(3, dtype=float),
                                  b=np.zeros(3, dtype=float),
                                  r=self.cutoff[0],
                                  safe=self.safe[0],
                                  rotation=self.rotation,
                                  angle_ptx=self.angle_ptx,
                                  gmax=0.,
                                  gmin=0.,
                                  shape=self.shape,
                                  centre=self.mx_centre)


        # 用于从odom中差分计算加速度
        self.last_time = None
        self.since_end_time = None
        self.last_linear_velocity = None
        print("初始化完成")

    def reset(self):
        '''重置规划器'''
        self.safe = np.zeros([self.path_len, self.shape[0], self.shape[1]], dtype=bool)  # 用于存储FFT滤波后的安全域
        self.rotation = None
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.linar_v = np.array([0.0, 0.0, 0.0])
        self.linar_a = np.array([0.0, 0.0, 0.0])  # 加速度，由于odom无输出，因此恒定为0
        self.target_direction = np.array([1.0, 0.0, 0.0])
        self.target_vector = np.array([10.0, 0., 0.])
        self.target_distance = 0.0
        self.target = np.array([10.0, 0., 0.])
        self.approaching = False  # 快要到达终点的标志位

        self.last_end_point = np.array([0.0, 0.0, 0.0])  # 上一个终点位置
        self.last_end_val = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])  # 上一个位置，IMU读数，首次规划为0

        # 用于从odom中差分计算加速度
        self.last_time = None
        self.since_end_time = None
        self.last_linear_velocity = None


    def spherical_distance(self, theta1, phi1, theta2, phi2):
        """
        计算球坐标系中两点之间的距离

        参数:
        r1, theta1, phi1 (float): 第一个点的球坐标（半径，极角θ，方位角φ，弧度单位）
        r2, theta2, phi2 (float): 第二个点的球坐标

        返回:
        float: 两点之间的欧氏距离
        """
        r = self.cutoff[0]
        delta_phi = phi1 - phi2
        cos_term = (np.sin(theta1) * np.sin(theta2) * np.cos(delta_phi) +
                    np.cos(theta1) * np.cos(theta2))
        distance_sq = 2 * r**2 * (1 - cos_term)
        return np.sqrt(max(distance_sq, 0))  # 避免负值因浮点误差

    def cal_sva(self, alpha, beta, gamma, a0, v0, p0, T):
        '''
        按相等时间间隔计算途径点位置、速度、加速度
        :param alpha:三轴参数 vec 3
        :param beta: 三轴参数 vec 3
        :param gamma: 三轴参数 vec 3
        :param a0: 这一段的初始加速度
        :param v0: 这一段的初始速度
        :param p0: 这一段的初始位置
        :param T: 这一段的初始预估时间
        :return:
        s:估计的路程，times:时间数组， p: 三轴位置，v:三轴速度数组，a:三轴加速度数组， tao:时间放缩因子数组，如果某一段不存在时间松弛则为1
        '''
        ki = round(T / self.kt)

        # 时间戳 - 绘图用
        times = np.zeros([ki + 1], dtype=float)
        times[0] = 0

        # 时间放缩因子，用于判断是否需要时间松弛
        tao = np.ones([ki + 1], dtype=float)

        # 位置
        p = np.zeros([ki + 1, 3], dtype=float)
        s = 0
        p[0] = p0

        # 速度
        v = np.zeros([ki + 1, 3], dtype=float)
        v[0] = v0

        vl = np.zeros([ki + 1], dtype=float)
        vl[0] = np.linalg.norm(v0)

        # 加速度
        a = np.zeros([ki + 1, 3], dtype=float)
        a[0] = a0

        al = np.zeros([ki + 1], dtype=float)
        al[0] = np.linalg.norm(a0)

        for i in range(1, ki+1):
            t = i / ki * T
            times[i] = t

            # 位置
            p[i] = np.array([
                alpha[0] * t ** 5 / 120 + beta[0] * t ** 4 / 24 + gamma[0] * t ** 3 / 6 + a0[0] * t ** 2 / 2 + v0[0] * t + p0[0],
                alpha[1] * t ** 5 / 120 + beta[1] * t ** 4 / 24 + gamma[1] * t ** 3 / 6 + a0[1] * t ** 2 / 2 + v0[1] * t + p0[1],
                alpha[2] * t ** 5 / 120 + beta[2] * t ** 4 / 24 + gamma[2] * t ** 3 / 6 + a0[2] * t ** 2 / 2 + v0[2] * t + p0[2],
            ])
            s += np.linalg.norm(p[i] - p[i-1])

            # 速度
            v[i] = np.array([
                alpha[0] * t ** 4 / 24 + beta[0] * t ** 3 / 6 + gamma[0] * t ** 2 / 2 + a0[0] * t + v0[0],
                alpha[1] * t ** 4 / 24 + beta[1] * t ** 3 / 6 + gamma[1] * t ** 2 / 2 + a0[1] * t + v0[1],
                alpha[2] * t ** 4 / 24 + beta[2] * t ** 3 / 6 + gamma[2] * t ** 2 / 2 + a0[2] * t + v0[2],
            ])
            vl[i] = np.linalg.norm(v[i])
            if vl[i] > self.max_spd:
                tao[i] = vl[i] / self.max_spd

            # 加速度
            a[i] = np.array([
                alpha[0] * t ** 3 / 6 + beta[0] * t ** 2 / 2 + gamma[0] * t + a0[0],
                alpha[1] * t ** 3 / 6 + beta[1] * t ** 2 / 2 + gamma[1] * t + a0[1],
                alpha[2] * t ** 3 / 6 + beta[2] * t ** 2 / 2 + gamma[2] * t + a0[2],
            ])
            al[i] = np.linalg.norm(a[i])
            if al[i] > self.max_acc:
                tao[i] = max(tao[i], np.sqrt(al[i] / self.max_acc))

        return s, times, p, v, a, tao


    def cal_sva_t(self, alpha, beta, gamma, a0, v0, p0, t):
        '''
        根据参数计算给定时刻的位置、速度、加速度
        :param alpha:三轴参数 vec 3
        :param beta: 三轴参数 vec 3
        :param gamma: 三轴参数 vec 3
        :param a0: 这一段的初始加速度
        :param v0: 这一段的初始速度
        :param p0: 这一段的初始位置
        :param T: 这一段的初始预估时间
        :return:
        p: 三轴位置，v:三轴速度数组，a:三轴加速度数组
        '''

        # 位置
        p = np.array([
            alpha[0] * t ** 5 / 120 + beta[0] * t ** 4 / 24 + gamma[0] * t ** 3 / 6 + a0[0] * t ** 2 / 2 + v0[0] * t + p0[0],
            alpha[1] * t ** 5 / 120 + beta[1] * t ** 4 / 24 + gamma[1] * t ** 3 / 6 + a0[1] * t ** 2 / 2 + v0[1] * t + p0[1],
            alpha[2] * t ** 5 / 120 + beta[2] * t ** 4 / 24 + gamma[2] * t ** 3 / 6 + a0[2] * t ** 2 / 2 + v0[2] * t + p0[2],
        ])

        # 速度
        v = np.array([
            alpha[0] * t ** 4 / 24 + beta[0] * t ** 3 / 6 + gamma[0] * t ** 2 / 2 + a0[0] * t + v0[0],
            alpha[1] * t ** 4 / 24 + beta[1] * t ** 3 / 6 + gamma[1] * t ** 2 / 2 + a0[1] * t + v0[1],
            alpha[2] * t ** 4 / 24 + beta[2] * t ** 3 / 6 + gamma[2] * t ** 2 / 2 + a0[2] * t + v0[2],
        ])


        # 加速度
        a = np.array([
            alpha[0] * t ** 3 / 6 + beta[0] * t ** 2 / 2 + gamma[0] * t + a0[0],
            alpha[1] * t ** 3 / 6 + beta[1] * t ** 2 / 2 + gamma[1] * t + a0[1],
            alpha[2] * t ** 3 / 6 + beta[2] * t ** 2 / 2 + gamma[2] * t + a0[2],
        ])

        return p, v, a

    def cal_best_T(self, alpha, beta, gamma, s, T):
        '''
        使用成本函数计算成本最低的轨迹耗时
        由于求解出的T并不完全和预期相同，因此不建议使用该函数迭代
        【计算错误，J中的alpha等参数也是T的隐函数】
        '''

        coefficients = [
            (1/5 * alpha[0]**2 + 1/5 * alpha[1]**2 + 1/5 * alpha[2]**2),
            (3/4 * alpha[0] * beta[0] + 3/4 * alpha[1] * beta[1] + 3/4 * alpha[2] * beta[2]),
            (2/3 * alpha[0] * gamma[0] + 2/3 * alpha[1] * gamma[1] + 2/3 * alpha[2] * gamma[2]) + (2/3 * beta[0]**2 + 2/3 * beta[1]**2 + 23 * beta[2]**2),
            beta[0] * gamma[0] + beta[1] * gamma[1] + beta[2] * gamma[2],
        ]

        roots = np.roots(coefficients)
        # real_roots = roots[np.isreal(roots)].real
        real_roots = roots.real

        minima_roots = []
        for root in real_roots:
            epsilon = 1e-5  # 一个小的增量值
            left_value = np.polyval(coefficients, (root - epsilon))
            right_value = np.polyval(coefficients, (root + epsilon))
            if left_value < 0 and right_value > 0 and root > s / self.max_spd:  # 限制极小值的范围
                minima_roots.append(root)

        if len(minima_roots) > 0:
            minima_roots = np.array(minima_roots)
            closest_root = minima_roots[np.argmin(np.abs(minima_roots - T))]
        else:
            closest_root = T
        return closest_root

    def cal_fully_defined_param(self, p0, v0, a0, pf, vf1, af, T):
        '''
        计算【完全定义】初值和终末值的轨迹
        用于时间松弛和接近目标点的情况
        :return: 三轴参数，alpha，beta，gamma， vec 3
        '''
        alpha = np.zeros(3, dtype=float)
        beta = np.zeros(3, dtype=float)
        gamma = np.zeros(3, dtype=float)

        for i in range(3):
            [alpha[i], beta[i], gamma[i]] = 1 / (T ** 5) * np.array([
                [720, -360 * T, 60 * T ** 2],
                [-360 * T, 168 * T ** 2, -24 * T ** 3],
                [60 * T ** 2, -24 * T ** 3, 3 * T ** 4]
            ]).dot(np.array([pf[i] - p0[i] - v0[i] * T - 1 / 2 * a0[i] * T ** 2,
                            vf1[i] - v0[i] - a0[i] * T,
                            af[i] - a0[i]]))

        return alpha, beta, gamma

    def cal_fully_defined_J(self, alpha, beta, gamma, T):
        '''
        计算【完全定义】的轨迹成本
        :return: 三轴参数，alpha，beta，gamma， vec 3
        '''
        J = np.zeros(3, dtype=float)
        for i in range(3):
            J[i] = gamma[i] ** 2 + beta[i] * gamma[i] * T + 1 / 3 * beta[i]**2 * T**2 + 1 / 3 * alpha[i] * gamma[i] * T**2 + \
                1 / 4 * alpha[i] * beta[i] * T**3 + 1 / 20 * alpha[i]**2 * T**4
        return np.sum(J)

    def cal_time_scaling(self, pl, vl, al, times, tao, t):
        '''
        根据时间缩放因子重新规划一段路径
        松弛产生的额外轨迹点会按照时间顺序插入原始轨迹并返回
        不建议使用，因为可能会造成预期之外的情况
        :param pl: 位置数组
        :param vl: 速度数组
        :param al: 加速度数组
        :param times: 时间数组
        :param tao: 时间放缩因子
        :param t: 进行缩放操作的时刻
        :return: 时间缩放后的时间，速度，加速度数组
        '''
        i = np.where(np.abs(times - t) < 1e-5)[0]
        if len(i) == 0 or i[0] == 0 or i[0] == len(times) - 1:  # 对首尾时刻不做处理
            return times, pl, vl, al

        i = i[0]

        T = (times[i + 1] - times[i - 1]) * tao
        alpha, beta, gamma = self.cal_fully_defined_param(pl[i - 1],
                                                          vl[i - 1],
                                                          al[i - 1],
                                                          pl[i + 1],
                                                          vl[i + 1],
                                                          al[i + 1], T)
        _, times_, pl_, v_, a_, _ = self.cal_sva(alpha, beta, gamma, al[i - 1], vl[i - 1], pl[i - 1], T)

        # 插入前的轨迹全保留
        # 插入的轨迹除去起始点，保留终止点（终止点理论上位置、速度、加速度都一样，但时间有所改变）
        # 插入后的轨迹的起始点被插入的轨迹替代
        t = np.concatenate([times[0:i], times_[1:len(times_)] + times[i - 1], times[i+2:len(times)]])
        p = np.concatenate([pl[0:i], pl_[1:len(pl_)], pl[i+2:len(pl)]])
        v = np.concatenate([vl[0:i], v_[1:len(v_)], vl[i+2:len(vl)]])
        a = np.concatenate([al[0:i], a_[1:len(a_)], al[i+2:len(al)]])

        return t, p, v, a

    def estimate_s(self, p1, v0, v1, T):
        '''
        估计路径长度
        :param s0: 截断点距离
        :param p1: 路径终点（本地为原点）
        :param v0: 初速度
        :param v1: 末速度
        :param T: 初步估计的时间
        :return: 路程，平均速率
        '''
        ax = -(v1[0] - v0[0]) / T + 3 * (p1[0] - v0[0] * T) / T ** 2
        ay = -(v1[1] - v0[1]) / T + 3 * (p1[1] - v0[1] * T) / T ** 2
        az = -(v1[2] - v0[2]) / T + 3 * (p1[2] - v0[2] * T) / T ** 2

        bx = (v1[0] - v0[0]) / T ** 2 - 2 * (p1[0] - v0[0] * T) / T ** 3
        by = (v1[1] - v0[1]) / T ** 2 - 2 * (p1[1] - v0[1] * T) / T ** 3
        bz = (v1[2] - v0[2]) / T ** 2 - 2 * (p1[2] - v0[2] * T) / T ** 3

        s = 0
        v = np.linalg.norm(v0)
        sx = [0]
        sy = [0]
        for i in range(10):
            t0 = i / 10 * T
            t1 = (i + 1) / 10 * T
            s += np.linalg.norm([
                bx * t1**3 + ax * t1**2 + v0[0] * t1 - (bx * t0**3 + ax * t0**2 + v0[0] * t0),
                by * t1 ** 3 + ay * t1 ** 2 + v0[1] * t1 - (by * t0 ** 3 + ay * t0 ** 2 + v0[1] * t0),
                bz * t1 ** 3 + az * t1 ** 2 + v0[2] * t1 - (bz * t0 ** 3 + az * t0 ** 2 + v0[2] * t0)
            ])
            v += np.linalg.norm([
                3 * bx * t1 ** 2 + 2 * ax * t1 + v0[0],
                3 * by * t1 ** 2 + 2 * ay * t1 + v0[1],
                3 * bz * t1 ** 2 + 2 * az * t1 + v0[2]
            ])
            sx.append(bx * t1**3 + ax * t1**2 + v0[0] * t1)
            sy.append(by * t1**3 + ay * t1**2 + v0[1] * t1)

        plt.figure()
        plt.plot(sx, sy, 'o-')
        plt.show()
        return s, v / 11

    def cal_forward_kinematics_divide(self, pl, vl, al, p1, times, tao, T1, T2, p1_safe=True):
        '''
        根据靶点推导正运动学
        作用：松弛时间，使轨迹可达
        以下均为【世界坐标系】
        :param pl: 位置数组(整段轨迹)
        :param vl: 速度数组
        :param al: 加速度数组
        :param p1: 靶点位置 vec 3
        :param times: 时间数组
        :param tao: 时间放缩因子
        :param T1:第一段用时
        :param T2:第二段用时
        :param p1_safe: 正推的途径点是否安全？若为false需要重新计算
        :return:松弛后的路径点pl, N*3
        '''
        p0 = pl[0]
        v0 = vl[0]
        a0 = al[0]

        # 最开始可以用T来做判断，并且只能用T做判断，因为可能存在靶点的移动，使用p1可能找不到
        idx1 = np.where(abs(times - T1) < 1e-6)[0][0]
        # idx1 = np.where(np.all(np.round(pl, 2) == np.round(p1, 2), axis=1))[0][0]
        # p1 = pl[idx1]
        v1 = vl[idx1]
        a1 = al[idx1]

        if T2 > 0:
            p2 = pl[-1]
            v2 = vl[-1]
            a2 = al[-1]
        else:  # 如果缺少第二段，则v1，a1均设为零
            v1 = np.zeros(3)
            a1 = np.zeros(3)

        # 重新推导正动力学得到pl，vl，al
        if not p1_safe:
            if T2 > 0:
                alpha1 = np.zeros(3, dtype=float)
                beta1 = np.zeros(3, dtype=float)
                gamma1 = np.zeros(3, dtype=float)

                alpha2 = np.zeros(3, dtype=float)
                beta2 = np.zeros(3, dtype=float)
                gamma2 = np.zeros(3, dtype=float)
                for i in range(3):
                    # (1)第一段 p1, v1给定，a1不限制
                    K1 = 1
                    K2 = -p0[i] - v0[i] * T1 - 0.5 * a0[i] * T1 ** 2
                    M1 = 0.
                    M2 = v1[i] - v0[i] - a0[i] * T1

                    kalpha1 = (1 / (T1 ** 5)) * (320 * K1 - 120 * M1 * T1)
                    balpha1 = (1 / (T1 ** 5)) * (320 * K2 - 120 * M2 * T1)
                    kbeta1 = (1 / (T1 ** 4)) * (-200 * K1 + 72 * M1 * T1)
                    bbeta1 = (1 / (T1 ** 4)) * (-200 * K2 + 72 * M2 * T1)
                    kgamma1 = (1 / (T1 ** 3)) * (40 * K1 - 12 * M1 * T1)
                    bgamma1 = (1 / (T1 ** 3)) * (40 * K2 - 12 * M2 * T1)

                    alpha1[i] = kalpha1 * p1[i] + balpha1
                    beta1[i] = kbeta1 * p1[i] + bbeta1
                    gamma1[i] = kgamma1 * p1[i] + bgamma1

                    if T2 > 0:  # 存在第二段
                        A = T1 ** 4 * kalpha1 / 24 + T1 ** 3 * kbeta1 / 6 + T1 ** 2 * kgamma1 / 2
                        B = T1 ** 4 * balpha1 / 24 + T1 ** 3 * bbeta1 / 6 + T1 ** 2 * bgamma1 / 2 + a0[i] * T1 + v0[i]
                        C = T1 ** 3 * kalpha1 / 6 + T1 ** 2 * kbeta1 / 2 + T1 * kgamma1
                        D = T1 ** 3 * balpha1 / 6 + T1 ** 2 * bbeta1 / 2 + T1 * bgamma1 + a0[i]

                        # (第二段) p2, v2固定，a2限制为0
                        K1 = -(1 + A * T2 + 0.5 * C * T2 ** 2)
                        K2 = p2[i] - B * T2 - 0.5 * D * T2 ** 2
                        M1 = -(A + C * T2)
                        M2 = (v2[i] - B - D * T2)
                        N1 = -C
                        N2 = -D

                        kalpha2 = (1 / T2 ** 5) * (720 * K1 - 360 * M1 * T2 + 60 * N1 * T2 ** 2)
                        balpha2 = (1 / T2 ** 5) * (720 * K2 - 360 * M2 * T2 + 60 * N2 * T2 ** 2)
                        kbeta2 = (1 / T2 ** 4) * (-360 * K1 + 168 * M1 * T2 - 24 * N1 * T2 ** 2)
                        bbeta2 = (1 / T2 ** 4) * (-360 * K2 + 168 * M2 * T2 - 24 * N2 * T2 ** 2)
                        kgamma2 = (1 / T2 ** 3) * (60 * K1 - 24 * M1 * T2 + 3 * N1 * T2 ** 2)
                        bgamma2 = (1 / T2 ** 3) * (60 * K2 - 24 * M2 * T2 + 3 * N2 * T2 ** 2)

                        alpha2[i] = kalpha2 * p1[i] + balpha2
                        beta2[i] = kbeta2 * p1[i] + bbeta2
                        gamma2[i] = kgamma2 * p1[i] + bgamma2
            else:
                T1 = 2 * np.linalg.norm(p1 - p0) / np.linalg.norm(v0) + self.kt
                alpha1, beta1, gamma1 = self.cal_fully_defined_param(p0, v0, a0, p1, v1, a1, T1)

            s1, times1, pl1, vl1, al1, tao1 = self.cal_sva(alpha1, beta1, gamma1, a0, v0, p0, T1)
            if T2 > 0:
                s2, times2, pl2, vl2, al2, tao2 = self.cal_sva(alpha2, beta2, gamma2, al1[-1], vl1[-1], p1, T2)
                pl = np.concatenate([pl1, pl2[1:]])
                times = np.concatenate([times1, times2[1:] + T1])
                vl = np.concatenate([vl1, vl2[1:]])
                al = np.concatenate([al1, al2[1:]])
                tao = np.concatenate([tao1, tao2[1:]])
                tao2 = tao2[1:]  # 不包括第一个截断点的末端
            else:
                pl = pl1
                times = times1
                vl = vl1
                al = al1
                tao = tao1
                tao2 = None

            # 重新计算p1,...
            idx1 = np.where(np.all(np.round(pl, 2) == np.round(p1, 2), axis=1))[0][0]
            p1 = pl[idx1]
            v1 = vl[idx1]
            a1 = al[idx1]

        else:
            tao1 = tao[0:idx1+1]
            tao2 = tao[idx1+1:] if T2 > 0 else None

        # # 记录每一次的迭代过程
        # if self.mode == 2:
        #     # 截断点为调整后的截断点
        #     plot_path_2D_k(pl.T, pf=p1, title='before iteration path', p_end=p2 if T2 > 0 else pl[-1],
        #                    target=self.target - self.p0, xlim=[min(-1, np.min(pl[..., 0])), self.cutoff[-1] + 1])
        #
        #     plt.figure(figsize=(8, 4))
        #
        #     plt.subplot(1, 2, 1)
        #     plt.plot(times, np.linalg.norm(vl, axis=1), '-o', label='velocity')
        #     plt.axvline(x=T1, color='r', linestyle='--')
        #     if np.max(tao) <= 1 + 1e-2:
        #         plt.ylim([0, self.max_spd + 1])
        #
        #     plt.subplot(1, 2, 2)
        #     plt.plot(times, np.linalg.norm(al, axis=1), '-o', label='acceleration')
        #     plt.axvline(x=T1, color='r', linestyle='--')
        #     if np.max(tao) <= 1 + 1e-2:
        #         plt.ylim([0, self.max_acc + 1])
        #     plt.title("before iteration")
        #     plt.show()
        #
        #     print("[before iteration] ", "max_tao1:{}, max_tao2:{}, T1:{}, T2:{}".format(
        #         np.max(tao1),
        #         np.max(tao2),
        #         T1, T2))

        # 通过时间二分法迭代
        for i in range(self.iter):  # 第0次迭代主要计算一个在安全域内的靶点
            if np.max(tao) <= 1 + 1e-2:
                break

            # 先检查第一段的放缩情况
            if np.max(tao1) > 1. + 1e-2:
                tao_t = np.max(tao1)  # 若使用np.max(tao)，则可能会求到tao2去
                idx_t = np.where(abs(tao - tao_t) < 1e-6)[0][0]

                # if idx_t != idx1:  # 如果异常点和截断点重合会发生错误
                p_t = pl[idx_t]
                v_t = vl[idx_t] / tao_t
                a_t = al[idx_t] / (tao_t**2)

                if idx_t == idx1:
                    v1 = v_t
                    a1 = a_t

                # R0t = np.linalg.norm(v0) ** 2 / self.max_acc
                # R0t = (0.5*np.linalg.norm(v0) + 0.5*np.linalg.norm(v_t)) ** 2 / self.max_acc

                # # === 老版本计算距离
                # R0t = max(np.linalg.norm(v0), np.linalg.norm(v_t)) ** 2 / self.max_acc
                # s0t = 2 * R0t * np.arcsin(np.linalg.norm(p_t - p0) / (2 * R0t))
                # if not s0t > 0:
                #     s0t = np.linalg.norm(p_t - p0) * 1.1

                # === 新版本计算距离
                c_0t = np.linalg.norm(p_t - p0)
                vec_0t = p_t - p0
                theta_0t = np.arccos(np.dot(v0, vec_0t) / (np.linalg.norm(v0) * np.linalg.norm(vec_0t)))
                s0t = c_0t * theta_0t / (2 * np.sin(theta_0t / 2) + 1e-6)
                if not 0 < s0t < c_0t * np.pi:
                    s0t = c_0t * 1.1

                T_0t = 2 * s0t / (np.linalg.norm(v0) + np.linalg.norm(v_t))

                # 最大tao之前的那一段轨迹
                alpha_0t, beta_0t, gamma_0t = self.cal_fully_defined_param(p0, v0, a0, p_t, v_t, a_t, T_0t)
                s0t_, times_0t, pl_0t, vl_0t, al_0t, tao_0t = self.cal_sva(alpha_0t, beta_0t, gamma_0t, a0, v0, p0, T_0t)

                # 最大tao到第一个途径点的轨迹
                # Rt1 = np.linalg.norm(v_t) ** 2 / self.max_acc
                # Rt1 = (0.5*np.linalg.norm(v_t) + 0.5*np.linalg.norm(v1)) ** 2 / self.max_acc

                # # === 老版本计算距离
                # Rt1 = max(np.linalg.norm(v_t), np.linalg.norm(v1)) ** 2 / self.max_acc
                # st1 = 2 * Rt1 * np.arcsin(np.linalg.norm(p1 - p_t) / (2 * Rt1))
                # if not st1 > 0:
                #     st1 = np.linalg.norm(p1 - p_t) * 1.1

                # === 新版本计算距离
                c_t1 = np.linalg.norm(p1 - p_t)
                vec_t1 = p1 - p_t
                theta_t1 = np.arccos(np.dot(v_t, vec_t1) / (np.linalg.norm(v_t) * np.linalg.norm(vec_t1)))
                st1 = c_t1 * theta_t1 / (2 * np.sin(theta_t1 / 2) + 1e-6)
                if not 0 < st1 < c_t1 * np.pi:
                    st1 = c_t1 * 1.1

                T_t1 = 2 * st1 / (np.linalg.norm(v_t) + np.linalg.norm(v1))
                alpha_t1, beta_t1, gamma_t1 = self.cal_fully_defined_param(p_t, v_t, a_t, p1, v1, a1, T_t1)
                st1_, times_t1, pl_t1, vl_t1, al_t1, tao_t1 = self.cal_sva(alpha_t1, beta_t1, gamma_t1, a_t, v_t, p_t, T_t1)

                # 将两段轨迹进行拼接
                pl1 = np.concatenate([pl_0t, pl_t1[1:]])
                times1 = np.concatenate([times_0t, times_t1[1:] + T_0t])
                vl1 = np.concatenate([vl_0t, vl_t1[1:]])
                al1 = np.concatenate([al_0t, al_t1[1:]])
                tao1 = np.concatenate([tao_0t, tao_t1[1:]])

                # 将计算得到的新轨迹替换原轨迹
                # 由于轨迹时间可能会发生拓展，因此拼接时需要将第二段轨迹的时间戳后移
                delta_t = times1[-1] - times1[0] - T1
                pl = np.concatenate([pl1, pl[idx1 + 1:]], axis=0)
                times = np.concatenate([times1, times[idx1 + 1:] + delta_t], axis=0)
                vl = np.concatenate([vl1, vl[idx1 + 1:]], axis=0)
                al = np.concatenate([al1, al[idx1 + 1:]], axis=0)
                tao = np.concatenate([tao1, tao[idx1 + 1:]], axis=0)

                idx1 = np.where(np.all(np.round(pl, 2) == np.round(p1, 2), axis=1))[0][0]
                T1 = times1[-1]
                tao1 = tao[0:idx1 + 1]
                tao2 = tao[idx1 + 1:] if T2 > 0 else None

            # 再检查第二段的放缩情况
            if tao2 is not None and len(tao2) > 0 and np.max(tao2) > 1. + 1e-2:
                tao_t = np.max(tao2)

                idx_t = np.where(abs(tao - tao_t) < 1e-6)[0][0]
                p_t = pl[idx_t]
                v_t = vl[idx_t] / tao_t
                a_t = al[idx_t] / (tao_t**2)

                # R1t = np.linalg.norm(v1) ** 2 / self.max_acc
                # R1t = (0.5*np.linalg.norm(v1) + 0.5*np.linalg.norm(v_t)) ** 2 / self.max_acc

                # # # === 老版本计算距离
                # R1t = max(np.linalg.norm(v1), np.linalg.norm(v_t)) ** 2 / self.max_acc
                # s1t = 2 * R1t * np.arcsin(np.linalg.norm(p_t - p1) / (2 * R1t))
                # if not s1t > 0:
                #     s1t = np.linalg.norm(p_t - p1) * 1.1

                # === 新版本计算距离
                c_1t = np.linalg.norm(p_t - p1)
                vec_1t = p_t - p1
                theta_1t = np.arccos(np.dot(v1, vec_1t) / (np.linalg.norm(v1) * np.linalg.norm(vec_1t)))
                s1t = c_1t * theta_1t / (2 * np.sin(theta_1t / 2) + 1e-6)
                if not 0 < s1t < c_1t * np.pi:
                    s1t = c_1t * 1.1

                T_1t = 2 * s1t / (np.linalg.norm(v1) + np.linalg.norm(v_t))

                # 最大tao之前的那一段轨迹
                alpha_1t, beta_1t, gamma_1t = self.cal_fully_defined_param(p1, v1, a1, p_t, v_t, a_t, T_1t)
                s1t_, times_1t, pl_1t, vl_1t, al_1t, tao_1t = self.cal_sva(alpha_1t, beta_1t, gamma_1t, a1, v1, p1, T_1t)

                # 最大tao到第一个途径点的轨迹
                # Rt2 = np.linalg.norm(v_t) ** 2 / self.max_acc
                # Rt2 = (0.5*np.linalg.norm(v_t) + 0.5*np.linalg.norm(2)) ** 2 / self.max_acc

                # # # === 老版本计算距离
                # Rt2 = max(np.linalg.norm(v_t), np.linalg.norm(v2)) ** 2 / self.max_acc
                # st2 = 2 * Rt2 * np.arcsin(np.linalg.norm(p2 - p_t) / (2 * Rt2))
                # if not st2 > 0:
                #     st2 = np.linalg.norm(p2 - p_t) * 1.1

                # === 新版本计算距离
                c_t2 = np.linalg.norm(p_t - p2)
                vec_t2 = p2 - p_t
                theta_t2 = np.arccos(np.dot(v_t, vec_t2) / (np.linalg.norm(v_t) * np.linalg.norm(vec_t2)))
                st2 = c_t2 * theta_t2 / (2 * np.sin(theta_t2 / 2) + 1e-6)
                if not 0 < st2 < c_t2 * np.pi:
                    st2 = c_t2 * 1.1


                T_t2 = 2 * st2 / (np.linalg.norm(v_t) + np.linalg.norm(v2))
                alpha_t2, beta_t2, gamma_t2 = self.cal_fully_defined_param(p_t, v_t, a_t, p2, v2, a2, T_t2)
                st2_, times_t2, pl_t2, vl_t2, al_t2, tao_t2 = self.cal_sva(alpha_t2, beta_t2, gamma_t2, a_t, v_t, p_t, T_t2)

                # 将两段轨迹进行拼接
                pl2 = np.concatenate([pl_1t, pl_t2[1:]])
                times2 = np.concatenate([times_1t, times_t2[1:] + T_1t])
                vl2 = np.concatenate([vl_1t, vl_t2[1:]])
                al2 = np.concatenate([al_1t, al_t2[1:]])
                tao2 = np.concatenate([tao_1t, tao_t2[1:]])

                # 将计算得到的新轨迹替换原轨迹
                pl = np.concatenate([pl[:idx1], pl2], axis=0)
                times = np.concatenate([times[:idx1], times2 + T1], axis=0)
                vl = np.concatenate([vl[:idx1], vl2], axis=0)
                al = np.concatenate([al[:idx1], al2], axis=0)
                tao = np.concatenate([tao[:idx1], tao2], axis=0)

                idx1 = np.where(np.all(np.round(pl, 2) == np.round(p1, 2), axis=1))[0][0]
                T2 = times2[-1] - times2[0]
                tao1 = tao[0:idx1 + 1]
                tao2 = tao[idx1 + 1:] if T2 > 0 else None


            # 记录每一次的迭代过程
            if self.mode == 2:
                print("[{}] ".format(i), "max_tao1:{}, max_tao2:{}, T1:{}, T2:{}".format(
                                                                      np.max(tao1),
                                                                      np.max(tao2),
                                                                      T1, T2))

            # # 记录每一次的迭代过程
            # if self.mode == 2:
            #     # 截断点为调整后的截断点
            #     plot_path_2D_k(pl.T, pf=p1, title='re-generated path-{}'.format(i), p_end=p2 if T2 > 0 else pl[-1],
            #                    target=self.target - self.p0, xlim=[-1, self.cutoff[-1] + 1])
            #
            #     plt.figure(figsize=(8, 4))
            #
            #     plt.subplot(1, 2, 1)
            #     plt.plot(times, np.linalg.norm(vl, axis=1), '-o', label='velocity')
            #     plt.axvline(x=T1, color='r', linestyle='--')
            #     if np.max(tao) <= 1 + 1e-2:
            #         plt.ylim([0, self.max_spd + 1])
            #
            #     plt.subplot(1, 2, 2)
            #     plt.plot(times, np.linalg.norm(al, axis=1), '-o', label='acceleration')
            #     plt.axvline(x=T1, color='r', linestyle='--')
            #     if np.max(tao) <= 1 + 1e-2:
            #         plt.ylim([0, self.max_acc + 1])
            #     plt.title("iteration {}".format(i))
            #     plt.show()
            #
            #     print("[{}] ".format(i), "max_tao1:{}, max_tao2:{}, T1:{}, T2:{}".format(
            #                                                           np.max(tao1),
            #                                                           np.max(tao2),
            #                                                           T1, T2))

        # 记录最后一次迭代
        if self.mode == 2 or self.mode == 3:
            # 截断点为调整后的截断点
            plot_path_2D_k(pl.T, pf=p1, title='re-generated path-{}'.format(i), p_end=p2 if T2 > 0 else pl[-1],
                           target=self.target_vector, xlim=[min(-1, np.min(pl[..., 0])), self.cutoff[-1] + 1])

            if self.mode == 2:
                plt.figure(figsize=(8, 4))

                plt.subplot(1, 2, 1)
                plt.plot(times, np.linalg.norm(vl, axis=1), '-o', label='velocity')
                plt.axvline(x=T1, color='r', linestyle='--')
                if np.max(tao) <= 1 + 1e-2:
                    plt.ylim([0, self.max_spd + 1])

                plt.subplot(1, 2, 2)
                plt.plot(times, np.linalg.norm(al, axis=1), '-o', label='acceleration')
                plt.axvline(x=T1, color='r', linestyle='--')
                if np.max(tao) <= 1 + 1e-2:
                    plt.ylim([0, self.max_acc + 1])
                plt.title("iteration {}".format(i))
                plt.show()

            print("[{}] ".format(i), "max_tao1:{}, max_tao2:{}, T1:{}, T2:{}".format(
                np.max(tao1),
                np.max(tao2),
                T1, T2))

        # # 迭代完成，开始绘图
        # if self.mode == 2:
        #     print("=====vl=====")
        #     print(np.round(vl, 1))
        #     print("==========")
        #
        #
        #     print("=====al=====")
        #     print(np.round(al, 1))
        #     print("==========")
        #
        #     print("=====t=====")
        #     print(np.round(times, 2))
        #     print("==========")

        return pl, vl, al, times

    def cal_best_idx_end(self, p0, v0, a0, p2, v2):
        '''
        计算截断距离最佳的目标靶点
        所有点、轨迹等均为【世界坐标系】
        :param p0: 无人机初始位置vec 3
        :param v0: 初始速度vec 3
        :param a0: 初始加速度vec 3
        # :param v1: 中间截断面处的速度约束 vec 3
        :param p2: 最后一个截断面的靶点位置 vec 3 需要转为世界坐标系输入。若远端规划失败，则填入-1
        :param v2: 最后一个截断面的速度限制 np.vec 3 需要转为世界坐标系输入。若远端规划失败，则填入-1
        # :param vt: 目标速率（标量）
        :return: 三轴目标方向 vec 3 世界坐标系下的位置
        '''
        v2_z = v2[2]
        s_all = 0.  # 储存全部路径长度，用于判断轨迹是否异常
        for iter in range(self.iter):  # 此处迭代次数太少太多都会出问题
            # 首次迭代，计算时间初值
            if iter == 0:
                if np.any(p2 == -1):  # 远端规划失效，均匀减速至0
                    # s1 = self.cutoff[0]
                    # T1 = np.sqrt(2 * s1 / self.max_acc)  # 最大速度刹车

                    # # === 老版本计算距离
                    # R1 = np.linalg.norm(v0) ** 2 / self.max_acc
                    # s1 = 2 * R1 * np.arcsin(self.cutoff[0] / (2 * R1))
                    # if not s1 > 0:
                    #     s1 = np.linalg.norm(self.cutoff[0]) * 1.1

                    # === 新版本计算距离
                    c_1 = self.cutoff[0]
                    e_p1 = v0 / np.linalg.norm(v0)  # 根据速度方向刹车，作为迭代初始值
                    theta_1 = np.arccos(np.dot(v0, e_p1) / (np.linalg.norm(v0) * np.linalg.norm(e_p1)))
                    s1 = c_1 * theta_1 / (2 * np.sin(theta_1 / 2) + 1e-6)
                    if not 0 < s1 < c_1 * np.pi:
                        s1 = c_1 * 1.1


                    T1 = 2 * s1 / np.linalg.norm(v0) + self.kt

                    T2 = 0.0
                    s2 = 0.0
                    tao2 = None
                    v1 = np.zeros(3)
                    # e_p1 = v0 / np.linalg.norm(v0)  # 根据速度方向刹车，作为迭代初始值
                else:  # 远端规划有效，最小成本为两个5次多项式轨迹合为一段轨迹，根据该性质计算最优速度和位置
                    if self.approaching:  # 接近目标时
                        # s1 = self.cutoff[0]

                        # # === 老版本计算距离
                        # R1 = np.linalg.norm(v0) ** 2 / self.max_acc
                        # s1 = 2 * R1 * np.arcsin(self.cutoff[0] / (2 * R1))
                        # if not s1 > 0:
                        #     s1 = np.linalg.norm(self.cutoff[0]) * 1.1

                        # === 新版本计算距离
                        c_1 = self.cutoff[0]
                        vec_1 = p2 - p0
                        theta_1 = np.arccos(np.dot(v0, vec_1) / (np.linalg.norm(v0) * np.linalg.norm(vec_1)))
                        s1 = c_1 * theta_1 / (2 * np.sin(theta_1 / 2) + 1e-6)
                        if not 0 < s1 < c_1 * np.pi:
                            s1 = c_1 * 1.1

                        s2 = self.target_distance - self.cutoff[0]
                        s2 = abs(s2)  # 当接近目标时，s2可能折返

                        V1 = np.sqrt(2 * self.max_acc * s2)

                        T2 = 2 * s2 / V1  # 匀减速运动
                        T1 = 2 * s1 / (V1 + np.linalg.norm(v0))

                    else:  # 正常状态
                        # s1 = self.cutoff[0]
                        # s2 = self.cutoff[1] - self.cutoff[0]

                        # # === 老版本计算距离
                        # # 按照圆周运动估计距离&时间
                        # R1 = np.linalg.norm(v0)**2 / self.max_acc
                        # R2 = np.linalg.norm(v2)**2 / self.max_acc
                        # s1 = 2 * R1 * np.arcsin(self.cutoff[0] / (2 * R1))
                        # s2 = 2 * R2 * np.arcsin((self.cutoff[1] - self.cutoff[0]) / (2 * R2))
                        # if not s1 > 0:
                        #     s1 = np.linalg.norm(self.cutoff[0]) * 1.1
                        # if not s2 > 0:
                        #     s2 = np.linalg.norm(self.cutoff[1] - self.cutoff[0]) * 1.1

                        # === 新版本计算距离
                        c_1 = self.cutoff[0]
                        c_2 = self.cutoff[1] - self.cutoff[0]
                        vec_1 = p2 - p0
                        theta_1 = np.arccos(np.dot(v0, vec_1) / (np.linalg.norm(v0) * np.linalg.norm(vec_1))) / 2
                        s1 = c_1 * theta_1 / (2 * np.sin(theta_1 / 2) + 1e-6)
                        if not 0 < s1 < c_1 * np.pi:
                            s1 = c_1 * 1.1
                        # vec_2 = p2 - p0
                        # theta_2 = np.arccos(np.dot(v2, vec_2) / (np.linalg.norm(v2) * np.linalg.norm(vec_2)))
                        theta_2 = theta_1
                        s2 = c_2 * theta_2 / (2 * np.sin(theta_2 / 2) + 1e-6)
                        if not 0 < s2 < c_2 * np.pi:
                            s2 = c_2 * 1.1

                        T1 = 2 * s1 / (self.exp_spd + np.linalg.norm(v0))
                        T2 = s2 / self.exp_spd

                    # 首次迭代时，计算中间途径点的初始值
                    alpha1, beta1, gamma1 = self.cal_fully_defined_param(np.zeros(3), self.linar_v, self.linar_a,
                                                                         p2, v2, np.zeros(3),
                                                                         T1 + T2)
                    p1, v1, _ = self.cal_sva_t(alpha1, beta1, gamma1, self.linar_a, self.linar_v, np.zeros(3), T1)
                    e_p1 = p1 / np.linalg.norm(p1)

                    v1 = compute_optimal_v1(p0, v0, a0, p1, p2, v2, np.zeros(3), T1, T2)
                    # pass

            # 后续迭代，计算时间迭代量
            else:
                # 全部安全，不再迭代
                # 若不加该语句，会导致过度迭代时速度放缩失效，造成震荡难收敛
                if np.max(tao) <= 1 + 0.1 or delta_dis < self.cutoff[0] * self.angle_ptx / np.sqrt(2):  # 这个函数只做粗略迭代
                    if self.mode:
                        print(">>>>>粗糙计算，迭代完成，次数{}<<<<<".format(iter))
                    break

                if np.any(p2 == -1):  # 远端规划失效，均匀减速至0
                    # T1 = np.sqrt(2 * s1 / self.max_acc)
                    T1 = 2 * s1 / np.linalg.norm(v0) + self.kt
                    T2 = 0.
                    s2 = 0.0
                    tao2 = None
                    e_p1 = p_sol / np.linalg.norm(p_sol)

                else:  # 正常状态 & 接近目标，迭代方法相同
                    '''根据最新情况计算优化问题的初始值'''
                    e_p1 = p_sol / np.linalg.norm(p_sol)

                    '''
                    TODO
                    经过测试，第一种在迭代后v1比较稳定，第二种由于v2的变化，v1会显著变化，导致随迭代不稳定
                    但实际上第二种在初步求解的结果更好
                    需要找到一种根本方法解决问题，迭代可能并不能解决本质问题
                    
                    可以尝试求解你动力学时只迭代2次，剩下都交给正动力学处理
                    【可用】在正动力学尝试使用二分法处理整段轨迹，保留所有轨迹点的位置，限制轨迹点的速度和加速度
                    
                    【难处理】此外，两段轨迹是耦合的，即便第二段不再超出要求，其第二段的时间、v2也会对第一段的末速度构成限制【难处理】
                    此外，求解优化问题时，可能会因如下原因求出严重的局部最优解：
                    安全域在xy方向存在缺失，但如果调整z轴方向，则可以找到一个xy平面内更好的解
                    由于梯度定义的限制，在寻找最优解的方向上会经过不安全的方向，因而无法通过梯度下降寻找到最优解
                    可能的解决方案：忽略z轴方向上的不安全情况，或减小z轴方向不安全域的权重，从而使优化函数更有机会找到xy平面内更安全的解
                    【可用】或者先延动力学最优的方向找最优解，再延安全域梯度方向归到安全域内
                    '''

                    # '''第一种：迭代计算最优的v1*，采用GPT方法，缺省p1'''
                    # if np.max(tao2) > 1.:  # 针对AB段
                    #     v2 /= max(np.max(tao2), 1.1)  # 避免迭代量太小造成收敛慢
                    #
                    # # ===== 【好】
                    # # 确保单调收敛，以防止迭代过程中出现震荡
                    # V1 = np.linalg.norm(v1)
                    # if np.max(tao1) > 1. + 1e-2:  # 针对OA段，兼顾AB段
                    #     V1 /= max(np.max(tao1), 1.1)  # 避免迭代量太小造成收敛慢
                    #
                    # T1 = 2 * s1 / (V1 + np.linalg.norm(v0))  # 根据实际规划出的速度迭代
                    # T2 = 2 * s2 / (V1 + np.linalg.norm(v2))  # 根据实际规划出的速度迭代
                    #
                    # T = T1 + T2
                    #
                    # # 计算 Δp, Δv, Δa
                    # delta_p = p2 - p0 - v0 * T - 0.5 * a0 * T ** 2
                    # delta_v = v2 - v0 - a0 * T
                    # delta_a = np.zeros(3) - a0
                    #
                    # # 五次多项式的系数 A, B, C
                    # A1 = (10 * delta_p - 4 * T * delta_v + 0.5 * T ** 2 * delta_a) / T ** 3
                    # B1 = (-15 * delta_p + 7 * T * delta_v - T ** 2 * delta_a) / T ** 4
                    # C1 = (6 * delta_p - 3 * T * delta_v + 0.5 * T ** 2 * delta_a) / T ** 5
                    #
                    # # 计算最优速度 v1*
                    # v1 = v0 + a0 * T1 + 3 * A1 * T1 ** 2 + 4 * B1 * T1 ** 3 + 5 * C1 * T1 ** 4

                    '''第二种：迭代计算最优的v1*，给定p1，采用五次多项式两段轨迹成本最小的方法'''
                    # 【目前采用这一种】
                    # T1 *= np.max(tao1)
                    # T2 *= np.max(tao2)

                    # T1 = 2 * s1 / (np.linalg.norm(v1) + np.linalg.norm(v0))  # 根据实际规划出的速度迭代
                    # T2 = 2 * s2 / (np.linalg.norm(v1) + np.linalg.norm(v2))  # 根据实际规划出的速度迭代

                    # 尝试同样迭代最优的v2
                    # 【重要】需要查看这一步操作后v2是否还需要/max tao
                    # print(v2)
                    v2 = (15 * (p2 - p_sol) - 7 * T2 * v1 - T2 ** 2 * a_sol) / (8 * T2)
                    v2[2] = v2_z
                    # print(v2)

                    # 储存上一次的时间，避免迭代出错提前跳出时T1出错
                    T1_ex = T1
                    T2_ex = T2

                    V0 = np.linalg.norm(v0)
                    V2 = np.linalg.norm(v2)

                    if np.max(tao2) > 1. + 1e-2:  # 针对AB段
                        v2 /= max(np.max(tao2), 1.1)  # 避免迭代量太小造成收敛慢

                    # ===== 【好】
                    # 确保单调收敛，以防止迭代过程中出现震荡
                    V1 = np.linalg.norm(v1)
                    if np.max(tao1) > 1. + 1e-2:  # 针对OA段，兼顾AB段
                        V1 /= max(np.max(tao1), 1.1)  # 避免迭代量太小造成收敛慢

                    T1 = 2 * s1 / (V1 + V0)  # 根据实际规划出的速度迭代
                    T2 = 2 * s2 / (V1 + V2)  # 根据实际规划出的速度迭代

                    v1 = compute_optimal_v1(p0, v0, a0, p_sol, p2, v2, np.zeros(3), T1, T2)  # / np.max(tao1) 没有效果，反而更差

                    if T1 > np.pi * V1 / self.max_acc \
                            or T2 > np.pi * V1 / self.max_acc \
                            or np.inner(self.target_vector, v1) < 0:
                        T1 = T1_ex
                        T2 = T2_ex
                        if self.mode:
                            print(">>>>>粗糙计算，路径异常提前跳出，次数{}<<<<<".format(iter))
                        break

                    # V1 = np.linalg.norm(v1)
                    # T1 = 2 * s1 / (V1 + np.linalg.norm(v0))  # 根据实际规划出的速度迭代
                    # T2 = 2 * s2 / (V1 + np.linalg.norm(v2))  # 根据实际规划出的速度迭代


                    # '''第三种：不对速度进行放缩，全靠后续时间放缩函数调整'''
                    # V1 = np.linalg.norm(v1)
                    # T1 = 2 * s1 / (V1 + np.linalg.norm(v0))  # 根据实际规划出的速度迭代
                    # T2 = 2 * s2 / (V1 + np.linalg.norm(v2))  # 根据实际规划出的速度迭代
                    # v1 = compute_optimal_v1(p0, v0, a0, p_sol, p2, v2, np.zeros(3), T1, T2)

            # 最终成本函数表达式的三轴系数，a为二次项，b为一次项，常数项忽略
            a = np.zeros(3, dtype=float)
            b = np.zeros(3, dtype=float)

            param_alpha = np.zeros([2, 3, 2], dtype=float)  # 两段，三轴，每段有a=kp+b两个参数
            param_beta = np.zeros([2, 3, 2], dtype=float)
            param_gamma = np.zeros([2, 3, 2], dtype=float)

            for idx in range(3):  # 三轴
                K1 = 1
                K2 = -p0[idx] - v0[idx] * T1 - 0.5 * a0[idx] * T1 ** 2
                M1 = 0.
                M2 = v1[idx] - v0[idx] - a0[idx] * T1

                kalpha1 = (1 / (T1 ** 5)) * (320 * K1 - 120 * M1 * T1)
                balpha1 = (1 / (T1 ** 5)) * (320 * K2 - 120 * M2 * T1)
                kbeta1 = (1 / (T1 ** 4)) * (-200 * K1 + 72 * M1 * T1)
                bbeta1 = (1 / (T1 ** 4)) * (-200 * K2 + 72 * M2 * T1)
                kgamma1 = (1 / (T1 ** 3)) * (40 * K1 - 12 * M1 * T1)
                bgamma1 = (1 / (T1 ** 3)) * (40 * K2 - 12 * M2 * T1)

                param_alpha[0][idx][0] = kalpha1
                param_alpha[0][idx][1] = balpha1
                param_beta[0][idx][0] = kbeta1
                param_beta[0][idx][1] = bbeta1
                param_gamma[0][idx][0] = kgamma1
                param_gamma[0][idx][1] = bgamma1

                a1, b1 = cal_ab(kalpha1, kbeta1, kgamma1, balpha1, bbeta1, bgamma1, T1)

                # 远端规划失效，均匀减速至0
                if np.any(p2 == -1):
                    a2, b2 = 0, 0
                else:
                    # (2) 第2步，计算从截断面到远端（下一个）靶点的成本
                    A = T1**4 * kalpha1 / 24 + T1**3 * kbeta1 / 6 + T1**2 * kgamma1 / 2
                    B = T1**4 * balpha1 / 24 + T1**3 * bbeta1 / 6 + T1**2 * bgamma1 / 2 + a0[idx] * T1 + v0[idx]
                    C = T1**3 * kalpha1 / 6 + T1**2 * kbeta1 / 2 + T1 * kgamma1
                    D = T1**3 * balpha1 / 6 + T1**2 * bbeta1 / 2 + T1 * bgamma1 + a0[idx]

                    # p2, v2固定，a2限制为0
                    K1 = -(1 + A * T2 + 0.5 * C * T2 ** 2)
                    K2 = p2[idx] - B * T2 - 0.5 * D * T2 ** 2
                    M1 = -(A + C * T2)
                    M2 = (v2[idx] - B - D * T2)
                    N1 = -C
                    N2 = -D

                    kalpha2 = (1 / T2 ** 5) * (720 * K1 - 360 * M1 * T2 + 60 * N1 * T2 ** 2)
                    balpha2 = (1 / T2 ** 5) * (720 * K2 - 360 * M2 * T2 + 60 * N2 * T2 ** 2)
                    kbeta2 = (1 / T2 ** 4) * (-360 * K1 + 168 * M1 * T2 - 24 * N1 * T2 ** 2)
                    bbeta2 = (1 / T2 ** 4) * (-360 * K2 + 168 * M2 * T2 - 24 * N2 * T2 ** 2)
                    kgamma2 = (1 / T2 ** 3) * (60 * K1 - 24 * M1 * T2 + 3 * N1 * T2 ** 2)
                    bgamma2 = (1 / T2 ** 3) * (60 * K2 - 24 * M2 * T2 + 3 * N2 * T2 ** 2)

                    param_alpha[1][idx][0] = kalpha2
                    param_alpha[1][idx][1] = balpha2
                    param_beta[1][idx][0] = kbeta2
                    param_beta[1][idx][1] = bbeta2
                    param_gamma[1][idx][0] = kgamma2
                    param_gamma[1][idx][1] = bgamma2

                    a2, b2 = cal_ab(kalpha2, kbeta2, kgamma2, balpha2, bbeta2, bgamma2, T2)

                    # A2 = T2 ** 4 * kalpha2 / 24 + T2 ** 3 * kbeta2 / 6 + T2 ** 2 * kgamma2 / 2
                    # B2 = T2 ** 4 * balpha2 / 24 + T2 ** 3 * bbeta2 / 6 + T2 ** 2 * bgamma2 / 2 + a0[idx ] * T1 + v0[idx]
                    # C2 = T2 ** 3 * kalpha2 / 6 + T2 ** 2 * kbeta2 / 2 + T2 * kgamma2
                    # D2 = T2 ** 3 * balpha2 / 6 + T2 ** 2 * bbeta2 / 2 + T2 * bgamma2 + a0[idx]

                a[idx] = a1 + a2
                b[idx] = b1 + b2

            # 根据优化条件计算最优目标点
            initial_p = [
                np.arccos(e_p1[2]),
                np.arctan2(e_p1[1], e_p1[0]),
                0.0
            ]

            self.solver.p0 = p0
            self.solver.a = a
            self.solver.b = b
            self.solver.safe = self.safe[0]

            self.solver.rotation = self.rotation

            # self.solver.penalty_debug_mx()
            # solver.dynamics_debug()

            # 局部坐标系
            mx, my = self.solver.solve(initial_p)

            angle_horizontal = self.angle_ptx * (self.mx_centre[1] - my)  # 水平角，左为正方向
            angle_vertical = np.pi / 2 - self.angle_ptx * (self.mx_centre[0] - mx)

            x_sol = self.cutoff[0] * np.sin(angle_vertical) * np.cos(angle_horizontal)
            y_sol = self.cutoff[0] * np.sin(angle_vertical) * np.sin(angle_horizontal)
            z_sol = self.cutoff[0] * np.cos(angle_vertical)

            # # 计算球坐标距离，判断是否跳出
            # delta_dis = self.spherical_distance(theta_sol, phi_sol, initial_p[0], initial_p[1])
            #
            # x_sol = self.cutoff[0] * np.sin(theta_sol) * np.cos(phi_sol)
            # y_sol = self.cutoff[0] * np.sin(theta_sol) * np.sin(phi_sol)
            # z_sol = self.cutoff[0] * np.cos(theta_sol)

            p_sol = self.rotation.apply([x_sol, y_sol, z_sol])  # 转换到世界坐标系
            if iter > 0:
                delta_dis = np.linalg.norm(p_sol - p_sol_ex)
            else:
                delta_dis = 1.0
            p_sol_ex = p_sol

            # if self.mode:
            #     print("世界坐标系下的最优解:", p_sol)

            # 更新迭代量
            # 计算三轴的参数
            alpha1 = np.sum(param_alpha[0] * np.array([p_sol, [1, 1, 1]]).T, axis=1)
            beta1 = np.sum(param_beta[0] * np.array([p_sol, [1, 1, 1]]).T, axis=1)
            gamma1 = np.sum(param_gamma[0] * np.array([p_sol, [1, 1, 1]]).T, axis=1)

            s1, times1, pl1, vl1, al1, tao1 = self.cal_sva(alpha1, beta1, gamma1, a0, v0, p0, T1)
            a_sol = np.array(al1[-1])

            # if self.mode:
            #     # print("最优点的速度", np.round(vl1[-1], 1), "加速度", np.round(al1[-1], 1))
            #     # 验证约束是否满足
            #     # print("\t球面约束项：:", (x_sol - p0[0]) ** 2 + (y_sol - p0[0]) ** 2 + (z_sol - p0[0]) ** 2 - self.cutoff[0] ** 2)
            #     print("\t安全域约束项：({}, {}):".format(round(mx), round(my)), self.solver.penalty_safe_mx(round(mx), round(my))-2)

            # if np.any(p2 == -1):  # 远端规划失效，均匀减速至0
            #     T1 = np.sqrt(2*s1 / self.max_x_acc)
            # else:
            #     # T1 = 2 * s1 / (vt + np.linalg.norm(v0))
            #     T1 = self.cal_best_T(alpha1, beta1, gamma1, s1, T1)

            if not np.any(p2 == -1):
                # 计算第二段路程
                alpha2 = np.sum(param_alpha[1] * np.array([p_sol, [1, 1, 1]]).T, axis=1)
                beta2 = np.sum(param_beta[1] * np.array([p_sol, [1, 1, 1]]).T, axis=1)
                gamma2 = np.sum(param_gamma[1] * np.array([p_sol, [1, 1, 1]]).T, axis=1)

                s2, times2, pl2, vl2, al2, tao2 = self.cal_sva(alpha2, beta2, gamma2, al1[-1], vl1[-1], p_sol, T2)

                # T2 = s2 / vt
                # T2 = self.cal_best_T(alpha2, beta2, gamma2, s2, T2)
                pl = np.concatenate([pl1, pl2[1:]])
                times = np.concatenate([times1, times2[1:] + T1])
                vl = np.concatenate([vl1, vl2[1:]])
                al = np.concatenate([al1, al2[1:]])
                tao = np.concatenate([tao1, tao2[1:]])
                s_all = s1 + s2
            else:
                pl = pl1
                times = times1
                vl = vl1
                al = al1
                tao = tao1
                s_all = s1

            if self.mode == 2:
                print("[{}] ".format(iter),
                      "p_sol:{}, is_safe:{}, delta_dis:{}, s1:{}, s2:{}, v1:{}, v2:{}, max_tao1:{}, max_tao2:{}, T1:{}, T2:{}".format(np.round(p_sol, 2),
                                                                                                  self.solver.penalty_safe_mx(round(mx),round(my)) - 2 < 0.1,
                                                                                                  np.round(delta_dis, 2),
                                                                                                  np.round(s1, 2), np.round(s2, 2),
                                                                                                  np.round(v1, 2), np.round(v2, 2),
                                                                                                  np.round(np.max(tao1), 2),
                                                                                                  np.round(np.max(tao2), 2),
                                                                                                  np.round(T1, 2), np.round(T2, 2)))

        if self.mode == 2 or self.mode == 3:
            plot_path_2D_k(pl.T, title='original path', pf=p_sol, target=self.target_vector, xlim=[min(-1, np.min(pl[..., 0])), self.cutoff[-1] + 1])

            if self.mode == 2:
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(times, np.linalg.norm(vl, axis=1), '-o', label='velocity')
                plt.axvline(x=T1, color='r', linestyle='--')
                if np.max(tao) <= 1 + 1e-2:
                    plt.ylim([0, self.max_spd + 1])

                plt.subplot(1, 2, 2)
                plt.plot(times, np.linalg.norm(al, axis=1), '-o', label='acceleration')
                plt.axvline(x=T1, color='r', linestyle='--')
                if np.max(tao) <= 1 + 1e-2:
                    plt.ylim([0, self.max_acc + 1])
                plt.title("iter:{}".format(iter))
                plt.show()

        return p_sol, pl, vl, al, times, T1, T2, v1, tao, s_all

    def find_nearest_one(self, matrix, ref_index, terminal_index):
        '''起始点至终止点的搜索'''
        if matrix[ref_index[0], ref_index[1]] == 1:  # 如果原本就安全，那就不需要找最近的安全点了
            return ref_index

        visited = set()
        queue = deque([ref_index])
        visited.add((ref_index[0],
                       ref_index[1]))

        kx = 1 if terminal_index[0] - ref_index[0] >= 0 else -1
        ky = 1 if terminal_index[1] - ref_index[1] >= 0 else -1

        x_min = min(terminal_index[0], ref_index[0])
        x_max = max(terminal_index[0], ref_index[0])
        y_min = min(terminal_index[1], ref_index[1])
        y_max = max(terminal_index[1], ref_index[1])

        while queue:
            x, y = queue.popleft()
            # 检查当前点是否为1，且在转向安全区域内
            if matrix[x, y] == 1:
                return np.array([x, y])

            # 添加相邻点到队列中
            # 单方向搜索
            for dx, dy in [(0, ky*self.ang_acc), (kx*self.ang_acc, 0)]:
                nx = x + dx
                ny = y + dy
                if (nx, ny) not in visited and \
                        y_min <= ny <= y_max and \
                        x_min <= nx <= x_max:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return terminal_index

    def find_nearest_margin(self, matrix, ref_index, terminal_index, r_safe, r_extreme):
        '''起始点至终止点的搜索，寻找距离目标点最近的安全域边界，步长为安全域半径'''
        if np.all(ref_index == terminal_index):
            return ref_index

        r_extreme = np.ceil(r_extreme)
        r_safe = round(r_safe)
        dis = max(np.linalg.norm(terminal_index - ref_index), 1.)  # 防止分母为0
        kx = (terminal_index[0] - ref_index[0]) / dis
        ky = (terminal_index[1] - ref_index[1]) / dis

        x_min = max(ref_index[0] - r_extreme, 0)
        x_max = min(ref_index[0] + r_extreme, self.shape[0] - 1)
        y_min = max(ref_index[1] - r_extreme, 0)
        y_max = min(ref_index[1] + r_extreme, self.shape[1] - 1)

        times = round(dis / r_safe)
        ex_visited = ref_index
        for ii in range(1, times+1):
            x = round(ref_index[0] + ii * kx * r_safe)
            y = round(ref_index[1] + ii * ky * r_safe)

            if x == terminal_index[0] and y == terminal_index[1]:  # 如果到达终点，则返回
                return np.array([x, y])

            if (matrix[x, y] == 0 or
                    x < x_min or x > x_max or
                    y < y_min or y > y_max):  # 如果进入不安全区域，或超出搜索范围，则返回上一个位于安全域内的点
                return ex_visited

            ex_visited = np.array([x, y])

        return ex_visited

    def find_nearest_one_origin(self, matrix, ref_index):
        '''有起始点，无终止点的搜索'''
        if np.any(ref_index == -1):
            return np.array([-1, -1])

        ref_index = map_to_edge(self.shape, (ref_index[0], ref_index[1]))

        if matrix[ref_index[0], ref_index[1]] == 1:  # 如果原本就安全，那就不需要找最近的安全点了
            return ref_index

        visited = set()
        queue = deque([ref_index])
        visited.add((ref_index[0],
                       ref_index[1]))

        x_min = 0.0
        x_max = self.shape[0] - 1
        y_min = 0.0
        y_max = self.shape[1] - 1

        # if self.mode and jidx == 0:
        #     print("角标：{}，速度：{}".format(jidx, np.linalg.norm(self.linar_v)))
        #     print("y轴-距离容限:[{},{}]".format(y_min, y_max))
        #     print("z轴-距离容限:[{},{}]\n".format(x_min, x_max))

        while queue:
            x, y = queue.popleft()
            # 检查当前点是否为1，且在转向安全区域内
            if matrix[x, y] == 1:
                return np.array([x, y])

            # 添加相邻点到队列中
            # 单方向搜索
            for dx, dy in [(0, self.ang_acc), (self.ang_acc, 0), (0, -self.ang_acc), (-self.ang_acc, 0)]:
                nx = x + dx
                ny = y + dy
                if (nx, ny) not in visited and \
                        y_min <= ny <= y_max and \
                        x_min <= nx <= x_max:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return np.array([-1, -1])

    def FFT_process(self, processed, img, j):
        # 对灰度图像按照保护距离和最远规划路径进行截断
        gray_image = np.clip(img,
                             0,
                             255 / self.dis_max * self.cutoff[j])  # 将距离限制在截断距离内  使用math.ceil会出现危险，使用math.floor会导致阈值无人能及

        if self.mode == 2 or self.mode == 3:
            plt.subplot(4, 2, j+1)
            plt.imshow(gray_image / 255 * self.dis_max)
            # plt.colorbar()

            # plt.title("{}m - cutoff".format(self.cutoff[j]))
            # plt.show()

        gray_image[..., 0] = 0  # 128维度的图片会造成左右循环，将边界值设为0避免认为边界安全
        gray_image[0, ...] = 0  # 32维度的图片会造成左右循环，将边界值设为0避免认为边界安全

        # 对灰度图求FFT -- freqz2方法
        freq1 = np.fft.fft2(gray_image, [self.Hshape, self.Hshape]) * self.H_D[j]
        filtered_image = np.abs(np.fft.ifft2(freq1))[0:self.shape[0], 0:self.shape[1]]
        print('-------------------------freq1--------------------------------------')
        print(freq1)
        print('---------------------------- ifft2 res------------------------------')
        print(np.fft.ifft2(freq1))
        if self.mode == 2 or self.mode == 3:
            plt.subplot(4, 2, j + 1 + 2)
            plt.imshow(filtered_image / 255 * self.dis_max)
            # plt.colorbar()

            # plt.title("{}m - cutoff".format(self.cutoff[j]))
            # plt.show()

        # 截断
        filtered_image = filtered_image >= (255 / self.dis_max * self.cutoff[j] - self.epsilon)  # 避免深度相机误差造成的安全域太小的情况
        # safe_area = np.copy(filtered_image)
        self.safe[j] = filtered_image

        if self.mode == 2 or self.mode == 3:
            plt.subplot(4, 2, j + 1 + 2 + 2)
            plt.imshow(filtered_image)
            # plt.colorbar()

        # ***************
        # 以下是滤波器求极值的部分
        # 仅对最远距离处求极值来获得空旷程度信息
        # ***************
        if j == 0:
            # 将矩阵描边，从而自然形成梯度 ！！【不可，这样会造成意外的零点，使优化器出错】
            # 要求：必须是滤波后从边界到安全域单调递增的值
            # 反例：如果 filtered_image.astype(np.int8) - 1  # 值域[-1-0]
            # 则当滤波器中心位于矩阵边缘时，由于滤波器框住的-1变少，值反而更靠近0
            filtered_image = filtered_image.astype(np.int8) + 1  # 值域[1-2]

            # filtered_image[0, :] = 0  # 第一行
            # filtered_image[-1, :] = 0  # 最后一行
            # filtered_image[:, 0] = 0  # 第一列
            # filtered_image[:, -1] = 0  # 最后一列

            freq1 = np.fft.fft2(filtered_image, [self.Hshape, self.Hshape])

            '''
            【！！】出错，此处self.H_S[j]滤波后的梯度在复杂情况下和朝向安全域的梯度不匹配
            原因：滤波器截止频率为图片长轴，导致时域覆盖了整幅图片
            如果截止频率为安全域尺度，则一方面梯度范围特别小，另一方面可能出现震荡，导致梯度方向错误
            需要找到尺度合适，边缘平坦的滤波器
            '''

            freq1 *= self.H_S[j]
            # freq1 *= self.H_D[j]
            # filtered_image = np.real(np.fft.ifft2(freq1))[0:self.shape[0], 0:self.shape[1]]
            filtered_image = np.abs(np.fft.ifft2(freq1))[0:self.shape[0], 0:self.shape[1]]

            self.solver.gradient = filtered_image
            self.solver.cal_gmax_gmin()
            if self.mode == 2 or self.mode == 3:
                plt.subplot(4, 2, j + 1 + 2 + 2 + 2)
                plt.imshow(filtered_image)
                # plt.colorbar()


        elif j == 1:
            # 对灰度图求FFT -- freqz2方法
            freq1 = np.fft.fft2(filtered_image, [self.Hshape, self.Hshape])

            freq1 *= self.H_S[j]
            filtered_image = np.abs(np.fft.ifft2(freq1))[0:self.shape[0], 0:self.shape[1]]

            dtsep = self.dtsep

            downsampled_image = filtered_image[::dtsep, ::dtsep]
            downsampled_safe_area = self.safe[j][::dtsep, ::dtsep]
            downsampled = downsampled_image*downsampled_safe_area

            # 使用numpy求极值
            greater0 = argrelextrema(downsampled, np.greater, axis=0, order=1)
            greater1 = argrelextrema(downsampled, np.greater, axis=1, order=1)

            set_max0 = set(zip(greater0[0]*dtsep, greater0[1]*dtsep))
            set_max1 = set(zip(greater1[0]*dtsep, greater1[1]*dtsep))

            tmp = np.array(list(set_max0 & set_max1)).T

            if self.mode == 2 or self.mode == 3:
                plt.subplot(4, 2, j+1+2+2+2)
                # 极大值画圈，用于调试
                # gimage = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
                gimage = (filtered_image * 255).astype(np.uint8)  # 将 filtered_image 转为可绘制的图像
                gimage = np.ascontiguousarray(gimage)  # 确保数组是连续的
                for idx in tmp.T:
                    cv2.circle(gimage, (idx[1], idx[0]), 5, 255, 2)

                plt.imshow(gimage)

                # plt.imshow(filtered_image)


            if len(tmp) == 0:
                return
            processed[j] = np.vstack((tmp, filtered_image[tuple(tmp)])).T

    def FFT_parallel(self, msg_lidar, msg_odom, target):
        '''
        输入lidar、odom话题数据，世界坐标系目标点，返回一条轨迹
        :param msg_lidar: lidar话题数据
        :param msg_odom: odom话题数据
        :param target:世界坐标系目标点 np.array vec 3
        # :param imgidx: ros时间戳, rospy.Time格式
        :return: 轨迹，N行 3列
        '''
        imgidx = msg_odom.header.stamp.to_sec()


        # since = time.time()
        # img = lidar_callback_Spherical(msg_lidar)
        # 首先判断姿态是否正常，若不正常，则直接返回
        # 【重要】初始姿态应该是水平的，尽可能朝向目标点
        self.rotation = R.from_quat([msg_odom.pose.pose.orientation.x,
                                     msg_odom.pose.pose.orientation.y,
                                     msg_odom.pose.pose.orientation.z,
                                     msg_odom.pose.pose.orientation.w])

        [roll, pitch, yaw] = self.rotation.as_euler('xyz', degrees=False)
        F_pitch = max(0, 1 - abs(pitch) / self.FOV[0])
        # F_roll = max(0, 1 - abs(roll) * (self.FOV[1]**2 + (max(0, self.FOV[0] - pitch))**2) / (2 * self.FOV[1] * max(0, self.FOV[0] - pitch)))
        # F_roll = max(0, 1 - (self.FOV[0] * max(0, 1 - abs(pitch) / self.FOV[0])) / self.FOV[1] * abs(roll))
        # alpha_s = np.arctan(self.FOV[0] / self.FOV[1])  # 水平视场角与垂直视场角的比值
        # alpha_l = np.arctan(self.FOV[1] / self.FOV[0])  # 垂直视场角与水平视场角的比值
        # if abs(roll) < alpha_s:
        #     A_roll = self.FOV[0] * self.FOV[1] - (self.FOV[0]**2 + self.FOV[1]**2) / 2 * np.tan(roll)
        # elif abs(roll) < alpha_l:
        #     A_roll = self.FOV[0] * (self.FOV[1] - self.FOV[0] * np.tan(roll))
        # else:
        #     A_roll = self.FOV[0]**2 * (1 - 1 / (np.tan(roll)))

        A_roll = self.FOV[0] * self.FOV[1] - ((1 - np.cos(roll)) / (np.sin(roll) * np.cos(roll))) * (self.FOV[0]**2 + self.FOV[1]**2 - 2 * self.FOV[0] * self.FOV[1] * np.cos(roll))
        F_roll = max(0, A_roll / (self.FOV[0] * self.FOV[1]))
        F_total = F_pitch * F_roll
        if F_total < 0.5:
            print("[{}]姿态不正常,roll: {}, pitch: {} ，丢弃数据".format(imgidx, roll * 180 / np.pi, pitch * 180 / np.pi))
            self.last_time = None
            self.last_linear_velocity = None
            self.last_end_point = np.array([0.0, 0.0, 0.0])
            self.last_end_val = np.array([0.0, 0.0, 0.0])
            self.last_position = np.array([0.0, 0.0, 0.0])
            return None, None, None, None, 2


        img = msg_lidar
        if self.mode == 2:
            print("================map id:{}================".format(imgidx))
            # plt.imshow(img)
            # # plt.axis("off");plt.savefig('original.png', bbox_inches='tight', dpi=300)
            # # plt.colorbar()
            # plt.show()

        if self.mode == 1:
            # print(">>>lidar数据处理用时(ms)：", round((time.time() - since) * 1000, 1))
            since = time.time()

        # wf -->local self.rotation.inv().apply(vec)
        # local --> wf self.rotation.apply(vec)

        self.p0 = np.array([msg_odom.pose.pose.position.x,
                            msg_odom.pose.pose.position.y,
                            msg_odom.pose.pose.position.z])  # 世界坐标系,原点在odom，仅用于计算目标点

        self.linar_v = np.array([msg_odom.twist.twist.linear.x,
                                msg_odom.twist.twist.linear.y,
                                msg_odom.twist.twist.linear.z])  # 线速度，世界坐标系

        # self.linar_a = np.zeros(3)
        if self.last_linear_velocity is not None:
            self.linar_a = (self.linar_v - self.last_linear_velocity) / (msg_odom.header.stamp.to_sec() - self.last_time)  # 线加速度，世界坐标系
        else:
            self.linar_a = np.zeros(3)

        # 计算从当前 UAV 位置到目标位置的方向向量
        self.target = target  # 世界坐标系，原点在odom
        self.target_vector = target - self.p0  # 全局坐标系
        self.target_distance = np.linalg.norm(self.target_vector)

        # 如果距离非零，计算单位方向向量
        if self.target_distance > 0.7:
            self.target_direction = self.target_vector / self.target_distance
        else:
            self.last_time = None
            self.last_linear_velocity = None
            self.last_end_point = np.array([0.0, 0.0, 0.0])
            self.last_end_val = np.array([0.0, 0.0, 0.0])
            self.last_position = np.array([0.0, 0.0, 0.0])
            return None, None, None, None, 0  # 到达目标点

        self.target_direction = self.rotation.inv().apply(self.target_direction)  # 【无人机】坐标系，单位向量，用于确定截断面目标点

        if self.mode:
            print("位置：", np.round(self.p0, 1))
            print("姿态rpy：", np.round(self.rotation.as_euler('xyz', degrees=True), 1))
            print("线速度：", np.round(self.linar_v, 1))
            print("线加速度：", np.round(self.linar_a, 1))
            print("目标方向单位向量:", np.round(self.target_direction, 2))

        if self.since_end_time is not None and abs(self.since_end_time - imgidx) > self.replan_time:
            self.since_end_time = None
            self.last_end_point = np.array([0, 0, 0], dtype=float)  # 重置远端靶点
            if self.mode:
                print(">>>重置目标点跟踪<<<")

        if np.any(self.last_end_point > 0):  # 不是第一次规划
            if self.since_end_time is None:
                self.since_end_time = imgidx  # 记录固定远端轨迹的时间

            end_point1 = self.last_end_point + self.last_end_val * (imgidx - self.last_time) \
                         - (self.p0 - self.last_position)  # 上一个目标点+速度偏移+起始点偏移

            # 检查坐标系
            po_local = self.rotation.inv().apply(end_point1)  # 转换至本地坐标系
            eo_local = po_local / np.linalg.norm(po_local)  # 最佳靶点的单位向量

            # 计算最佳方向的投影
            angle_horizontal1 = np.arctan(eo_local[1] / eo_local[0])  # phi
            refoy = (self.mx_centre[1] - angle_horizontal1 / self.angle_ptx)  # y

            angle_vertical1 = np.arccos(eo_local[2])  # theta
            refox = (self.mx_centre[0] + (angle_vertical1 - np.pi / 2) / self.angle_ptx)  # x

            ref_endpt = np.array([round(refox), round(refoy)], dtype=int)  # 上一次规划的远端目标在当前时刻的方向
            ref_endpt = map_to_edge(self.shape, ref_endpt)

            # if self.mode:
            #     print(" >>>跟踪上时刻目标点<<< ")
        else:
            ref_endpt = None
            if self.mode:
                print(">>>重新计算远端目标<<<")


        # p0 = np.array([0, 0, 0], dtype=float)  # 为计算简便，将局部坐标系建立在无人机位置

        # 根据参考方向计算参考靶点
        try:
            ref = np.array([round(self.mx_centre[0] + (np.arccos(self.target_direction[2]) - np.pi / 2) / self.angle_ptx),
                            round(self.mx_centre[1] - np.sign(self.target_direction[0]) * np.arctan(self.target_direction[1] / self.target_direction[0]) / self.angle_ptx)],
                           dtype=int)
        except:
            print("参考方向计算错误，强制直行")
            ref = self.mx_centre

        if self.mode:
            print("目标方向靶点:", ref)

        # 将参考点限制在观测范围内
        # ref = np.array([
        #     np.clip(ref[0], 0, self.shape[0] - 1),
        #     np.clip(ref[1], 0, self.shape[1] - 1)
        # ], dtype=int)
        ref = map_to_edge(self.shape, ref)

        # 用于存储FFT滤波后的安全方向
        # 格式：截断距离个数*极值个数*[矩阵角标x，矩阵角标y,极值数值]
        # 注意此处矩阵角标为float型
        if self.mode == 1:
            print(">>>初始化用时(ms)：", round((time.time()-since)*1000, 1))
            since = time.time()

        processed = [[] for _ in range(self.path_len)]
        jobs = []

        # 并行运行
        if self.mode == 0 or self.mode == 1:
            for j in range(self.path_len):
                p = threading.Thread(target=self.FFT_process, args=(processed,
                                                               img,
                                                               j))
                p.start()
                jobs.append(p)

            for job in jobs:
                job.join()

        elif self.mode == 2 or self.mode == 3:
            plt.figure(figsize=(10, 10))
            for j in range(self.path_len):
                self.FFT_process(processed,
                                 img,
                                 j)
            plt.show()
            plt.figure(figsize=(12, 5))

        if self.mode == 1:
            print(">>>FFT滤波用时(ms)：", round((time.time()-since)*1000, 1))
            since = time.time()


        '''需要根据FFT滤波安全域情况计算'''
        '''若不存在安全域，则直接跳转至异常处理部分'''
        # 即将到达终点
        # 由于滤波器能观测的安全距离恒定，因此只判断一次
        if (self.target_distance < self.cutoff[1] and self.safe[1][ref[0], ref[1]] == 1):
            self.approaching = True
            if self.mode:
                print("------------------[无人机接近目标<{}m]------------------------".format(self.cutoff[1]))

        # 最近的一个截断面直达终点，直接快速计算终点路径
        if (self.target_distance < self.cutoff[0] and self.safe[0][ref[0], ref[1]] == 1):
            if self.mode:
                print("------------------[无人机接近目标<{}m]------------------------".format(self.cutoff[1]))

            # # === 老版本计算距离
            # R1 = np.linalg.norm(self.linar_v) ** 2 / self.max_acc
            # s = 2 * R1 * np.arcsin(self.target_distance / (2 * R1))
            # if not s > 0:
            #     s = self.target_distance

            # === 新版本计算距离
            c = self.target_distance
            theta = np.arccos(np.dot(self.linar_v, self.target_vector) / (np.linalg.norm(self.linar_v) * np.linalg.norm(self.target_vector)))
            s = c * theta / (2 * np.sin(theta / 2) + 1e-6)
            if not 0 < s < c * np.pi:
                s = c * 1.1

            T = np.sqrt(2 * s / self.max_acc)

            # for i in range(self.iter + 1):
            #     alpha, beta, gamma = self.cal_fully_defined_param(np.zeros(3), self.linar_v, self.linar_a, target_vector,
            #                                  np.zeros(3), np.zeros(3), T)
            #
            #     s, times, p, v, a, tao = self.cal_sva(alpha, beta, gamma,
            #                                           self.linar_a, self.linar_v, np.zeros(3), T)
            #     if np.max(tao) > 1 + 1e-2:
            #         T *= np.max(tao)
            #     else:
            #         break

            alpha, beta, gamma = self.cal_fully_defined_param(np.zeros(3), self.linar_v, self.linar_a, self.target_vector,
                                         np.zeros(3), np.zeros(3), T)

            s, times, p, v, a, tao = self.cal_sva(alpha, beta, gamma,
                                                  self.linar_a, self.linar_v, np.zeros(3), T)

            if np.max(tao) > 1 + 1e-2:
                p, v, a, times = self.cal_forward_kinematics_divide(p, v, a, p[-1], times, tao, times[-1], -1, True)

            if self.mode == 2:
                plot_path_2D_k(p.T, title='approaching path', pf=self.target_vector, target=self.target_vector)

                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(times, np.linalg.norm(v, axis=1), '-o', label='velocity')

                plt.subplot(1, 2, 2)
                plt.plot(times, np.linalg.norm(a, axis=1), '-o', label='acceleration')
                plt.show()

                # print("=====vl=====")
                # print(np.round(v, 1))
                # print("==========")
                #
                # print("=====al=====")
                # print(np.round(a, 1))
                # print("==========")
                #
                # print("=====t=====")
                # print(np.round(times, 2))
                # print("==========")

            return p, v, a, times, 1

        if np.max(self.safe[0]) < 0.01:
            deep_flag = 0
        else:
            deep_flag = 2
            ref_index = np.ones([self.path_len, 2], dtype=int)
            ref_index_theory = np.ones([self.path_len, 2], dtype=int)
            pf_all_wf = np.zeros([2, 3])
            if not self.approaching:
                # 强制要求跟踪点同时在两个安全域内，避免速度过快时发生危险
                if ref_endpt is not None and self.safe[1][ref_endpt[0], ref_endpt[1]] and self.safe[0][ref_endpt[0], ref_endpt[1]]:
                    ref_index_theory[1] = ref_endpt
                    if self.mode:
                        print(" >>>跟踪上时刻目标点<<< ")
                elif self.safe[1][ref[0], ref[1]] and self.safe[0][ref[0], ref[1]]:  # 如果目标点就在安全域内，则直接朝向目标点，不再执行极值点搜索
                    ref_index_theory[1] = ref
                    if self.mode:
                        print(">>>重新计算远端目标--目标方向安全<<<")
                else:
                    if self.mode:
                        print(">>>重新计算远端目标--根据极值筛选<<<")
                    # ***最远的截断距离：目标方向和空旷程度的加权值
                    all_possible_lines = []
                    all_costs = []
                    if len(processed[1]) > 0:
                        c_max = np.max(processed[1], axis=0)[2]
                        if c_max > 1.0:  # 限制理论上限，消除旁瓣的影响
                            c_max = 1.0
                        if ref_endpt is not None:
                            self.w_costs[1][3] = 0.5 * (1 - abs(self.since_end_time - imgidx) / self.replan_time)  # 时间衰减权重
                            self.w_costs[1] = self.w_costs[1] / np.sum(self.w_costs[1])
                        for mx, my, c in processed[1]:
                            # if not self.safe[1][round(mx), round(my)]:  # 滤波时已滤除过
                            #     continue

                            # (1) 动力学成本
                            angle_horizontal = self.angle_ptx * (self.mx_centre[1] - my)
                            angle_vertical = np.pi / 2 - self.angle_ptx * (self.mx_centre[0] - mx)
                            xx = self.cutoff[1] * np.sin(angle_vertical) * np.cos(angle_horizontal)
                            yy = self.cutoff[1] * np.sin(angle_vertical) * np.sin(angle_horizontal)
                            zz = self.cutoff[1] * np.cos(angle_vertical)

                            p2_hat = self.rotation.apply([xx, yy, zz])  # 转换到世界坐标系
                            v2_hat = p2_hat

                            theta_hat = np.arccos(self.linar_v @ v2_hat / (np.linalg.norm(self.linar_v) * np.linalg.norm(v2_hat)))
                            s_hat = self.cutoff[1] * theta_hat / (2 * np.sin(theta_hat / 2))
                            T_hat = 2 * s_hat / (self.exp_spd + np.linalg.norm(self.linar_v))

                            # s_hat, v_bar = self.estimate_s(p2_hat, self.linar_v, v2_hat, T_hat)
                            #
                            # T_hat = s_hat / v_bar
                            alpha2, beta2, gamma2 = self.cal_fully_defined_param(np.zeros(3), self.linar_v, self.linar_a,
                                                                                    p2_hat, v2_hat, np.zeros(3), T_hat)
                            J_hat = self.cal_fully_defined_J(alpha2, beta2, gamma2, T_hat)


                            # (2) 空旷成本
                            '''
                            【严重】
                            如果滤波器太小，极值点变成上下两排，就会导致空旷成本出现异常，进而造成选择异常
                            
                            此外，经过最大值归一化后，空旷成本是差异最大的成本
                            需要调整其计算方式，并降低其权重
                            '''
                            # k = 2 * np.linalg.norm(self.mx_centre) / c_max
                            # cost2 = k * (c_max - c) + 1  # 避免c=0造成计算结果异常
                            if c >= 1.0:
                                cost2 = 1e-6
                            else:  # 使用最大值归一化
                                cost2 = c_max - c


                            # (3) 参考方向成本
                            cost3 = np.linalg.norm([ref[0] - mx, ref[1] - my])

                            # (4) 上时刻规划方向成本
                            if ref_endpt is not None:
                                # cost4 = np.log2(np.linalg.norm([ref_endpt[0] - mx, ref_endpt[1] - my])) + 1e-6
                                cost4 = np.linalg.norm([ref_endpt[0] - mx, ref_endpt[1] - my])
                            else:
                                cost4 = 1.

                            all_possible_lines.append([mx, my])
                            all_costs.append([J_hat, cost2, cost3, cost4])

                    if len(all_possible_lines) == 0:  # 没有找到任何符合要求的轨迹点，则停止规划
                        ref_index_theory[1] = [-1, -1]

                    elif len(all_possible_lines) > 1:
                        # 计算当前截断距离下的最佳点位
                        all_possible_lines = np.array(all_possible_lines)
                        all_costs = np.array(all_costs)

                        # debug用
                        # tmp = np.concatenate([all_possible_lines, all_costs], axis=1)
                        # sorted_indices = np.lexsort(tmp.T[::-1])  # 反转列顺序后传入
                        # sorted_arr = tmp[sorted_indices]
                        # np.set_printoptions(suppress=True); print(np.round(sorted_arr, 2))

                        # if self.mode:
                        #     print("每条路径的各类成本（空旷成本、参考方向成本）：\n",
                        #           all_costs / np.max(all_costs, axis=0))

                        quantile_sums = all_costs / np.max(all_costs, axis=0) @ self.w_costs[1]

                        best_idx = np.argmin(quantile_sums)

                        ref_index_theory[1] = all_possible_lines[best_idx]
                    else:
                        ref_index_theory[1] = all_possible_lines[0]

                if np.any(ref_index_theory[1] == -1):  # 最远距离没有安全方向，可能是一堵墙，因此方向暂时不做处理，标记深度以便刹车
                    deep_flag = 1
                else:
                    # 尽可能寻找无需抬升机身的方向，沿矩阵的竖线靠近矩阵的赤道
                    # 【有误，这样会导致无人机下跌至地面】
                    # ref_middle_line = np.array([self.mx_centre[0], ref_index_theory[1][1]])
                    # ref_index[1] = self.find_nearest_one(self.safe[1], ref_middle_line, ref_index_theory[1])

                    # # 之前可用
                    # ref_index[1] = ref_index_theory[1]

                    # 详细寻找
                    ref_index[1] = self.find_nearest_margin(self.safe[1], ref_index_theory[1], ref,
                                                            r_safe=self.r[1],
                                                            r_extreme=self.shape[1] / 2 / 9)


                # *** 【新版本】计算后两个截断距离的方向，采用类似A*的启发式成本
                # *** 注意，该方法只适用于LiDAR感知，并且仅适合有位置传感器的硬件
                if not np.any(ref_index[1] == -1):  # 远端规划成功
                    # 计算本地坐标系下的下一个目标点
                    angle_horizontal = self.angle_ptx * (self.mx_centre[1] - ref_index[1][1])  # 水平角，左为正方向
                    angle_vertical = np.pi / 2 - self.angle_ptx * (self.mx_centre[0] - ref_index[1][0])

                    xx = self.cutoff[1] * np.sin(angle_vertical) * np.cos(angle_horizontal)
                    yy = self.cutoff[1] * np.sin(angle_vertical) * np.sin(angle_horizontal)
                    zz = self.cutoff[1] * np.cos(angle_vertical)
                    pf2_local = np.array([xx, yy, zz], dtype=float)

                    # # 远端距离的z轴高度直接设置成目标高度
                    # # 【出错！！这是局部坐标系！这样会使无人机往下栽，最终导致轨迹出错】
                    # pf2_local = np.array([xx, yy, self.target[2]], dtype=float)
                    # pf2_local = pf2_local / np.linalg.norm(pf2_local) * self.cutoff[1]  # 幅度调整至截断面的球上

                    # 计算世界坐标系下的目标点，坐标原点为无人机本地
                    pf2 = self.rotation.apply(pf2_local)  # 世界坐标系，原点为无人机位置
                    pf_all_wf[1] = pf2

                    # 计算末端速度
                    future_target = self.target_vector - pf2  # 世界坐标系，原点为odom
                    future_dis = np.linalg.norm(future_target)
                    # future_dir = future_target / future_dis

                    # v2 朝向目标方向
                    # v2 = future_dir * min(self.exp_spd, np.sqrt(2 * self.max_acc * future_dis)) \
                    #     if self.target_distance > self.cutoff[-1] else np.zeros(3)  # 世界坐标系

                    # v2 朝向起始点指向远端截断距离的方向
                    v2 = pf2 / np.linalg.norm(pf2) * min(self.exp_spd, np.sqrt(2 * self.max_acc * future_dis)) \
                        if self.target_distance > self.cutoff[-1] else np.zeros(3)

                else:  # 远端规划失败
                    pf2 = np.array([-1,-1,-1])
                    v2 = np.zeros(3)
            else:  # 接近目标，将远端规划设置为目标点
                pf2 = self.target_vector
                pf_all_wf[1] = pf2
                ref_index[1] = ref
                ref_index_theory[1] = ref
                v2 = np.zeros(3)

            if self.mode == 1:
                print(">>>远端规划用时(ms)：", round((time.time() - since) * 1000, 1))
                since = time.time()

            # 计算最佳靶点
            # 此处初始p0, v0, a0选取的均为无人机当前位姿状态
            # 计算过程为：从当前位姿→目标靶点→下一个靶点
            p_ref, pl, vl, al, times, T1, T2, v1, tao, s_all = self.cal_best_idx_end(np.zeros(3), self.linar_v, self.linar_a, pf2, v2)
            pf_all_wf[0] = p_ref  # 世界坐标系，原点为无人机位置

            # v2 = vl[-1] / np.linalg.norm(vl[-1]) * self.exp_spd  # 将终末速度调整至期望速度，但方向不变

            # 将最佳靶点转换至本地坐标系
            p_ref_local = self.rotation.inv().apply(p_ref)  # 本地坐标系
            e_ref_local = p_ref_local / np.linalg.norm(p_ref_local)  # 最佳靶点的单位向量

            # 计算最佳方向的投影
            angle_horizontal1 = np.arctan(e_ref_local[1] / e_ref_local[0])  # phi
            ref_index_theory[0][1] = self.mx_centre[1] - angle_horizontal1 / self.angle_ptx  # y

            angle_vertical1 = np.arccos(e_ref_local[2])  # theta
            ref_index_theory[0][0] = self.mx_centre[0] + (angle_vertical1 - np.pi / 2) / self.angle_ptx  # x

            # print(ref_index_theory[0])
            ref_index[0] = self.find_nearest_one_origin(self.safe[0], ref_index_theory[0])
            err_ptx = np.linalg.norm(ref_index[0] - ref_index_theory[0])  # 求解出的最优解和安全域内解的像素距离-ptx

            if self.mode == 2 or self.mode == -1:
                print(">>>>> 理想靶点：", ref_index_theory[0], "实际靶点：", ref_index[0], ">>>>>像素距离：", err_ptx)

                refox = round(self.solver.refox)
                refoy = round(self.solver.refoy)

                plt.figure()
                gimage = np.array(self.safe[0]*255, dtype=np.uint8)
                cv2.circle(gimage, (ref_index_theory[0][1], ref_index_theory[0][0]), 8, 200, 2)  # 理想靶点
                cv2.circle(gimage, (ref_index[0][1], ref_index[0][0]), 4, 100, 2)  # 实际靶点
                cv2.circle(gimage, (refoy, refox), 4, 50, 2)  # 运动学靶点
                plt.imshow(gimage)
                plt.title("{}-comparison: ideal(Green), real(Blue), dynamics(purple) ".format(imgidx))
                if self.mode == -1:
                    plt.savefig("img/{}-cmp-{}px.png".format(imgidx, np.linalg.norm(ref_index[0] - ref_index_theory[0])))
                else:
                    plt.show()

            if np.any(ref_index[0] == -1):  # 本次规划失败
                deep_flag = 0

            if self.mode == 1:
                print(">>>近端规划用时(ms)：", round((time.time()-since)*1000, 1))
                since = time.time()


        '''1  没有任何安全空间，以最快速度刹车'''
        if deep_flag == 0:
            print("当前安全域为空集，需要刹车")
            self.last_time = None
            self.last_linear_velocity = None
            self.last_end_point = np.array([0.0, 0.0, 0.0])
            self.last_end_val = np.array([0.0, 0.0, 0.0])
            self.last_position = np.array([0.0, 0.0, 0.0])
            return None, None, None, None, 0

        if s_all > self.cutoff[deep_flag-1] * np.pi / 2:  # 检查轨迹是否过于曲折或出现折返
            print("当前轨迹规划可能异常，需要刹车")
            self.last_time = None
            self.last_linear_velocity = None
            self.last_end_point = np.array([0.0, 0.0, 0.0])
            self.last_end_val = np.array([0.0, 0.0, 0.0])
            self.last_position = np.array([0.0, 0.0, 0.0])
            return None, None, None, None, 0

        '''2.  存在完整或部分路径'''
        # 如果求解结果偏离安全域，则根据安全域内的靶点推正运动学求截断点，否则沿用最初的求解点
        if err_ptx:
            # 先计算纠正到安全域内的靶点位置
            angle_horizontal = self.angle_ptx * (self.mx_centre[1] - ref_index[0][1])  # 水平角，左为正方向
            angle_vertical = np.pi / 2 - self.angle_ptx * (self.mx_centre[0] - ref_index[0][0])

            xx = self.cutoff[0] * np.sin(angle_vertical) * np.cos(angle_horizontal)
            yy = self.cutoff[0] * np.sin(angle_vertical) * np.sin(angle_horizontal)
            zz = self.cutoff[0] * np.cos(angle_vertical)

            p1_wf = self.rotation.apply([xx, yy, zz])  # 转换到世界坐标系
        else:
            p1_wf = p_ref

        '''3. 根据截断点迭代，使速度、加速度满足无人机最大性能的要求'''
        # 两个情况：中间点不再安全域内、时间需要缩放
        if err_ptx or np.max(tao) > 1.:
            if deep_flag == self.path_len:  # 截断面都安全
                # pl, vl, al, times = self.cal_forward_kinematics_end(np.zeros(3), self.linar_v, self.linar_a,
                #                                                     p1_wf, v1, pf_all_wf[1], v2, T1, T2)

                pl, vl, al, times = self.cal_forward_kinematics_divide(pl, vl, al, p1_wf, times, tao, T1, T2, err_ptx==0)
            else:  # 规划距离小于最大截断距离
                # pl, vl, al, times = self.cal_forward_kinematics_end(np.zeros(3), self.linar_v, self.linar_a,
                #                                                     p1_wf, v1, [-1, -1, -1], np.zeros(3), T1, -1)

                pl, vl, al, times = self.cal_forward_kinematics_divide(pl, vl, al, p1_wf, times, tao, T1, -1, err_ptx==0)
                print("后续安全距离{}为空集，需要刹车：".format(self.cutoff[deep_flag]))

        # if self.mode == 2:
        #     plot_path_2D_k(pl.T, pf=pf_all_wf[0], title='re-generated path', target=self.target_direction * self.target_distance, xlim=[-1, self.cutoff[-1] + 1])

        if self.mode == 2:
            print("\n参考角标：\n", ref_index,
                    # "\n轨迹点：\n", np.round(pl, 1)
                  )

        if self.mode == 1:
            print(">>>收尾用时(ms)：", round((time.time() - since) * 1000, 1))

        # 用于从odom中差分计算加速度
        self.last_time = imgidx
        self.last_linear_velocity = self.linar_v
        self.last_end_point = np.array(pl[-1])
        self.last_end_val = np.array(vl[-1])
        self.last_position = self.p0
        return pl, vl, al, times, deep_flag

def convert_to_depth_image(msg):
    '''将128*1024线的lidar深度数据转下采样到32*128'''
    # 根据字节序确定数据类型
    dtype = np.dtype('>u2') if msg.is_bigendian else np.dtype('<u2')
    # 将二进制数据转换为uint16数组
    depth_flat = np.frombuffer(msg.data, dtype=dtype)
    # 计算每行的元素数（每元素占2字节）
    elements_per_row = msg.step // 2
    # 重塑为二维数组并提取有效宽度
    depth_image = depth_flat.reshape(msg.height, elements_per_row)[:, :msg.width]

    depth_image = np.clip(depth_image, 0, 10 * 1000)  # 深度范围限制在10m内，单位为mm
    depth_image = depth_image / (10 * 1000) * 255  # 归一化到0-255  128*1024

    # 按2的指数采样，极限压缩分辨率 32x128
    depth_image = depth_image[::4, ::4]
    depth_image = depth_image[:, 64:192]  # 截取有效区域 128

    # # 压缩分辨率 45*360
    # depth_image = cv2.resize(
    #     depth_image,
    #     (360, 45),  # OpenCV尺寸顺序为(width, height)
    #     interpolation=cv2.INTER_NEAREST)
    # depth_image = depth_image[:, 90:270]  # 截取有效区域 45x180

    # # 大分辨率
    # depth_image = cv2.resize(
    #     depth_image,
    #     (720, 90),  # OpenCV尺寸顺序为(width, height)
    #     interpolation=cv2.INTER_NEAREST)
    # depth_image = depth_image[:, 180:540]  # 截取有效区域

    depth_image[depth_image == 0] = np.max(depth_image)  # 0值设为最大值，避免噪点造成的滤波错误

    return depth_image



if __name__ == '__main__':
    total_time = 0.0
    cnt = 0
    fft_net = FFT_plan(2)  #-1 绘制对比图 #0 静默模式 #1 文字输出 #2 plot debug  # 默认为0，仿真时应当设为0或1

    # start_time = rospy.Time(secs=1742545084+40)  # 1~30  20,50有问题， 40求解器求不到安全域内

    msg_lidar = None
    msg_odom = None
    # with rosbag.Bag('lidar_msg/2025-03-21-16-18-04.bag', 'r') as lidar_msg:
    #     for topic, msg, t in lidar_msg.read_messages(start_time=start_time):
    #         if cnt % 50 == 0:
    #             print(cnt)
    #
    #         if cnt % 100 == 0 and cnt != 0:
    #             break
    #
    #         if topic == '/ouster/range_image':
    #
    #             since = time.time()
    #             msg_lidar = convert_to_depth_image(msg)
    #
    #             print(">>>转换用时(ms)：", round((time.time() - since) * 1000, 1))  # 1~2ms
    #
    #
    #
    #         elif topic == '/Odometry':
    #             msg_odom = msg
    #
    #
    #         if msg_lidar is not None and msg_odom is not None:
    #             since = time.time()
    #             fft_net.FFT_parallel(msg_lidar, msg_odom, np.array([60.0, -2.0, 1.5]), cnt)  #  60.0, -2.0, 1.5
    #             # 远端到达目标测试8. + 22., 0.4 - 8.8, 1.9 + 0.9， t=40
    #             # 近端到达目标测试4. + 22., 0.08 - 8.8, 1.1 + 0.9， t=40
    #             # 近端出错测试5. + 22., -0.08 - 8.8, 1.1 + 0.9, t=40
    #
    #             total_time += time.time() - since
    #             cnt += 1
    #             break
    #             #
    #             msg_lidar = None
    #             msg_odom = None

    jmp_cnt = 0
    start_time = rospy.Time(secs=732)  # 2_auto_only_tree.bag  271后四个严重抖动
    # start_time = rospy.Time(secs=682 + 49 + 3)  # lidar_msg/0522/2025-05-22-13-49-10.bag
    # start_time = rospy.Time(secs=857+20+2+2)  # 881.019出现轨迹折返 lidar_msg/remote_ctrl_1/2025-05-15-17-02-47.bag
    # start_time = rospy.Time(secs=536)  # 881.019出现轨迹折返 lidar_msg/remote_ctrl_1/2025-05-15-17-02-47.bag
    with rosbag.Bag('/home/jack/2025-05-22-17-59-20.bag', 'r') as lidar_msg:
        for topic, msg, t in lidar_msg.read_messages(start_time=start_time):
            if jmp_cnt > 0:
                jmp_cnt = (jmp_cnt + 1) % (11)  # 每0.1*4秒判断一次
                continue

            if cnt % 50 == 0 and cnt != 0:
                print(cnt)

            if cnt % 100 == 0 and cnt != 0:
                break

            if topic == '/os0_cloud_node/range_image':

                since = time.time()
                msg_lidar = convert_to_depth_image(msg)

                print(">>>转换用时(ms)：", round((time.time() - since) * 1000, 1))  # 1~2ms



            elif topic == '/robot/dlio/odom_node/odom':
                msg_odom = msg


            if msg_lidar is not None and msg_odom is not None:
                since = time.time()
                fft_net.FFT_parallel(msg_lidar, msg_odom, np.array([200., 2., 1.5]))  #  60.0, -2.0, 1.5
                # 远端到达目标测试8. + 22., 0.4 - 8.8, 1.9 + 0.9， t=40
                # 近端到达目标测试4. + 22., 0.08 - 8.8, 1.1 + 0.9， t=40
                # 近端出错测试5. + 22., -0.08 - 8.8, 1.1 + 0.9, t=40

                total_time += time.time() - since
                cnt += 1
                jmp_cnt += 1
                break
                #
                msg_lidar = None
                msg_odom = None


    print("总用时mm：", round(total_time * 1000, 2))
    print("总次数：", cnt)
    print("平均用时mm：", round(total_time * 1000 / cnt, 2))


# 通过惩罚项幅度匹配，使其能够尽可能平衡动力学成本【完成】影响不大
# 远端截断距离处的点是不是要尽可能靠近矩阵中线（避免z轴方向上的动力学成本）【解决】将远端距离的z轴直接设置成目标点高度
# 尝试松弛约束项，或对末端速度/加速度约束进行调整【有用，调整速度约束】
# 最后尝试能否建立联合优化方程，避免某一时刻加速度太大无法跟踪


'''
TODO OLD
尝试将中间是速度换成三次多项式的最小值【经过尝试，效果也不好，不如五次多项式，需要检查推导过程】
检查更多的时刻，评估时间松弛的效果
验证惩罚项函数的赋值（观察迭代过程中各个成本的变化）【除边界惩罚外，数值匹配】
尝试降低优化问题求解精度
尝试换用更多cv的方式压缩分辨率
https://docs.opencv.org/4.5.1/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
'''

'''
【完成】尝试将2截断面的梯度也绘制进去，以便调试
【完成】继续调试到达目标/远端失效等情况
调试更换滤波器后的原始情况
【完成，可能造成结果差异很大】尝试梯度更正后降低求解器精度的情况
【效果很差】尝试梯度更正后使用原来的求解器的情况
【完成】调整第一次迭代的次数
【求解精度不如以前，主要是cv重采样容易出现噪点】调试下采样图片后的情况
【完成】尝试降低求解器精度


【提醒】
求解靶点时，多迭代很容易出现轨迹的严重问题，尽可能把加速度松弛放到后面的函数中，第一个函数只保证计算精度即可
'''

'''
TODO 0526
（1）【完成】当目标方向在远端安全域内时，可以设置直接朝向目标方向前进
如果不在，则再计算极值等？

（2）远端极值提取滤波器不应该太大，否则角度误差太大，轨迹会出现异常
是否还需要求极值操作？求极值的意义何在？
----
滤波器的作用是先选一个粗的备选点
因此尝试使用一个更大的滤波器
找到一个合适的极值点
然后从极值点到目标方向执行广度优先搜索
搜索区域<滤波器半径
步长近似为安全域半径
直至找到安全区域的边界
----

（3）【完成】判断轨迹异常情况，若异常（如：来不及转向等，需要借助实际运行情况找一个合适的指标）则刹车

'''

