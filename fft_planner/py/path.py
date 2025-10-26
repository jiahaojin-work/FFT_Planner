import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_path(data, target=None, title=''):
    # return

    # 创建一个新的3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将data数组的每一行分别赋值给x, y, z
    x, y, z = data
    length = np.linalg.norm([x[-1], y[-1], z[-1]], 2)

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)
    z = np.insert(z, 0, 0)

    # 绘制轨迹点，用 '.' 表示
    ax.plot(x, y, z, marker=".", markersize=10, color="blue")

    # 绘制起始点，用 'o' 表示
    ax.scatter(x[0], y[0], z[0], color='green', label='Start', s=50)

    # 绘制终点，用 'x' 表示
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=50)

    if target is not None:
        ax.plot([0, target[0]*length], [0, target[1]*length], [0, target[2]*length], color="red", alpha=0.5, marker='o')

    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # # 设置三个轴的刻度尺度相同
    # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    # mid_x = (x.max() + x.min()) * 0.5
    # # mid_y = (y.max() + y.min()) * 0.5
    # # mid_z = (z.max() + z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # # ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlim(-1, 11)

    # 添加图例
    ax.legend()

    plt.title(title)

    # 显示图形
    plt.show()

def plot_path_k(data, target=None, title=''):
    # return

    # 创建一个新的3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将data数组的每一行分别赋值给x, y, z
    x, y, z = data
    length = np.linalg.norm([x[-1], y[-1], z[-1]], 2)

    # x = np.insert(x, 0, 0)
    # y = np.insert(y, 0, 0)
    # z = np.insert(z, 0, 0)

    # 绘制轨迹点，用 '.' 表示
    ax.plot(x, y, z, marker=".", markersize=10, color="blue")

    # 绘制起始点，用 'o' 表示
    ax.scatter(x[0], y[0], z[0], color='green', label='Start', s=50)

    # 绘制终点，用 'x' 表示
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=50)

    if target is not None:
        ax.plot([0, target[0]*length], [0, target[1]*length], [0, target[2]*length], color="red", alpha=0.5, marker='o')

    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # # 设置三个轴的刻度尺度相同
    # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    # mid_x = (x.max() + x.min()) * 0.5
    # # mid_y = (y.max() + y.min()) * 0.5
    # # mid_z = (z.max() + z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # # ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlim(-1, 11)

    # 添加图例
    ax.legend()

    plt.title(title)

    # 显示图形
    plt.show()

def plot_path_2D(data, target=None, title=''):
    # 创建一个新的3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 将data数组的每一行分别赋值给x, y, z
    x, y, _ = data

    length = np.linalg.norm([x[-1], y[-1]], 2)

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    # 绘制轨迹点，用 '.' 表示
    ax.plot(x, y, marker=".", markersize=5, color="blue", alpha=0.7)

    # 绘制起始点，用 'o' 表示
    ax.scatter(x[0], y[0], color='green', label='Start', s=100)

    # 绘制终点，用 'x' 表示
    ax.scatter(x[-1], y[-1], color='red', label='End', s=100)

    if target is not None:
        ax.plot([0, target[0] * length], [0, target[1] * length], color="red", alpha=0.5, marker='o')

    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # 设置三个轴的刻度尺度相同
    max_range_x = np.array([x.max() - x.min()]).max() / 2.0
    max_range_y = np.array([y.max() - y.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    # mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range_x - 1, mid_x + max_range_x + 1)
    ax.set_ylim(mid_y - max_range_y - 1, mid_y + max_range_y + 1)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_ylim(-3, 3)

    # 添加图例
    ax.legend()

    plt.title(title)

    # 显示图形
    plt.show()


def plot_path_2D_k(data, target=None, title='', color='blue', pf=None, xlim=None, ylim=None, p_end=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(211)

    # 将data数组的每一行分别赋值给x, y, z
    x, y, z = data

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    # 绘制轨迹点，用 '.' 表示
    ax.plot(x, y, marker=".", markersize=5, color=color, alpha=0.5)

    # 绘制起始点，用 'o' 表示
    ax.scatter(x[0], y[0], color='green', label='Start', s=100)

    # 绘制终点，用 'x' 表示
    if p_end is not None:
        ax.scatter(p_end[0], p_end[1], color='red', label='End', s=100)
    else:
        ax.scatter(x[-1], y[-1], color='red', label='End', s=100)

    if target is not None:
        length = np.linalg.norm(target, 2)
        length = min(length, 10)
        target = np.array(target) * length / np.linalg.norm(np.array(target))
        ax.plot([0, target[0]], [0, target[1]], color="red", alpha=0.5, marker='o', label='target dir')

    if pf is not None:
        ax.scatter(pf[0], pf[1], marker='o', s=100, color='orange', label='cutoff point')

    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    if xlim is None:
        max_range_x = np.array([x.max() - x.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        ax.set_xlim(mid_x - max_range_x - 1, mid_x + max_range_x + 1)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is None:
        max_range_y = np.array([y.max() - y.min()]).max() / 2.0
        mid_y = (y.max() + y.min()) * 0.5
        ax.set_ylim(mid_y - max_range_y - 1, mid_y + max_range_y + 1)
    else:
        ax.set_ylim(ylim[0], ylim[1])

    # ax.set_ylim(-3, 3)

    # 添加图例
    ax.legend()
    plt.title(title)

    ax2 = fig.add_subplot(212)
    xx = np.arange(len(z))
    plt.plot(xx, z, "o-", label='height')

    # 设置轴标签
    ax2.set_xlabel('points')
    ax2.set_ylabel('Z axis')

    ax2.set_ylim(-2., 2.)
    ax2.legend()

    # 显示图形
    plt.show()

def plot_path_2D_continued(ax, data, target=None, color='blue'):
    # 显示图形
    plt.figure()

    # 将data数组的每一行分别赋值给x, y, z
    x, y, _ = data

    length = np.linalg.norm([x[-1], y[-1]], 2)

    # x = np.insert(x, 0, 0)
    # y = np.insert(y, 0, 0)

    # 绘制轨迹点，用 '.' 表示
    ax.plot(x, y, marker=".", markersize=5, color=color, alpha=0.5)

    # 绘制起始点，用 'o' 表示
    ax.scatter(x[0], y[0], color='green', label='Start', s=100)

    # 绘制终点，用 'x' 表示
    ax.scatter(x[-1], y[-1], color='red', label='End', s=100)

    if target is not None:
        ax.plot([0, target[0] * length], [0, target[1] * length], color="red", alpha=0.5, marker='o')

    # 设置轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # 设置三个轴的刻度尺度相同
    max_range_x = np.array([x.max() - x.min()]).max() / 2.0
    max_range_y = np.array([y.max() - y.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    # mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range_x - 1, mid_x + max_range_x + 1)
    ax.set_ylim(mid_y - max_range_y - 1, mid_y + max_range_y + 1)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_ylim(-3, 3)

    # 添加图例
    ax.legend()


if __name__=='__main__':

    path = np.array([[0.06177026, 0.19050497, 0.5681139, 1.2004638, 2.0688007, 3.1052742, 4.163741, 5.1453514, 6.121377, 7.065019],
                    [-0.30907536, -0.5425281, -0.5738906, -0.38530105, -0.03375396, 0.31250995, 0.46657103, 0.34704465, 0.05615011, -0.31546935],
                    [0.16050619, 0.2837682, 0.27041674, 0.11046106, -0.1601218, -0.45273483, -0.6442069, -0.68149805, -0.61996484, -0.48601735]])

    plot_path(path)

    dis = []
    dis.append(np.linalg.norm(np.zeros(3) - path.T[0],
                              2))
    for i in range(path.shape[1] - 1):
        dis.append(np.linalg.norm(path.T[i] - path.T[i+1],
                       2))


    print(dis)

