import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
# 画三维方向箭头用
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


# 获取数据
def get_data(relative_path, file_name):
    with open(os.path.join(relative_path, file_name), "rb") as f:
        data = pickle.load(f)
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return x, y, z

# 主函数
relative_path = '../results'
for i in range(1, 18):
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # friend trajectory
    x, y, z = get_data(relative_path, 'friend_pos_{}.pkl'.format(i))
    for i in range(1, int(len(x)/10)):
        direct = np.array([x[10*i]-x[10*i-1], y[10*i]-y[10*i-1], z[10*i]-z[10*i-1]])
        direct_norm = direct / np.linalg.norm(direct)
        # print(direct)
        # ax.scatter(x[10*i], y[10*i], z[10*i], color='r', marker=(direct, 0))
        ax.quiver(x[10*i], y[10*i], z[10*i], direct[0], direct[1], direct[2], color='g', length=1, arrow_length_ratio=3)

    l1, = ax.plot(x, y, z, color='r')
    # for ii in range(1, x.shape[0]):
    #     ax.arrow3D(x[ii - 1], y[ii - 1], z[ii - 1], (x[ii] - x[ii - 1]) / 2,
    #                (y[ii] - y[ii - 1]) / 2, (z[ii] - z[ii - 1]) / 2)

    # enemy trajectory
    x, y, z = get_data(relative_path, 'enemy_pos_{}.pkl'.format(i))
    # ax.scatter(x, y, z, color='g', marker='o')
    l2, = ax.plot(x, y, z, color='g')

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.legend(handles=[l1, l2, ], labels=['Friend', 'Enemy'], loc='best')
    plt.show()
