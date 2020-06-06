import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def rotate(x, y, z, b, c, a):
    rotatez = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0,0,1]])
    rotatex = np.array([[1,0,0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rotatey = np.array([[math.cos(c), 0, math.sin(c)], [0,1,0], [-math.sin(c), 0, math.cos(c)]])
    R = np.dot(np.dot(rotatex,rotatey),rotatez)
    x1 = x*R[0][0] + y*R[0][1] + z*R[0][2]
    y1 = x*R[1][0] + y*R[1][1] + z*R[1][2]
    z1 = x*R[2][0] + y*R[2][1] + z*R[2][2]
    return [x1, y1, z1]

def update(num, pz, pc, pconst):
    ax = plt.axes(projection='3d')
    # Setting the axes properties
    ax.set_xlim3d([-4.0, 4.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-4.0, 4.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-4.0, 4.0])
    ax.set_zlabel('Z')
    ax.set_title('3D Experiment')

    ps = 0.05
    rx = math.pi/30
    ry = math.pi/30
    rz = math.pi/30
    pz = rotate(pz[0],pz[1],pz[2], rx*num, ry*0, rz*0)
    pc = rotate(pc[0],pc[1],pc[2], rx*num, ry*0, rz*0)
    ax.plot_surface(pz[0], pz[1], pz[2]+num*ps, color='r')
    ax.plot_surface(pc[0], pc[1], pc[2]+num*ps, color='lightyellow')
    ax.plot_surface(pconst[0], pconst[1], pconst[2], color='navy')
    return ax

# 1. 申请fig和坐标ax
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 2. 生成图形对象的所有数据（极坐标转为三维直角坐标）

# 两个极坐标角度
u = np.linspace(0, 2 * np.pi, 50)  # linspace的功能用来创建等差数列
v = np.linspace(0, np.pi/4, 50)
xz = 0.8 * np.outer(np.cos(u), np.sin(v))  # outer（a，b） 外积：a的每个元素乘以b的每个元素，二维数组
yz = 0.8 * np.outer(np.sin(u), np.sin(v))
zz = np.sqrt(xz ** 2 + yz ** 2)  # 圆锥体的高
xz, yz, zz = rotate(xz, yz, zz, 0, 0, 0) # 初始旋转

# 生成球数据
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(np.pi/4, np.pi, 100)
xc = 0.8 * np.outer(np.cos(u), np.sin(v))
yc = 0.8 * np.outer(np.sin(u), np.sin(v))
zc = 0.8 * np.outer(np.ones(np.size(u)), np.cos(v))

xconst = 1.5 * np.outer(np.cos(u), np.sin(v)) + 2
yconst = 1.5 * np.outer(np.sin(u), np.sin(v)) + 2
zconst = 1.5 * np.outer(np.ones(np.size(u)), np.cos(v)) + 2

# 3. 给定坐标属性
ax.set_xlim3d([-4.0, 4.0])
ax.set_xlabel('X')
ax.set_ylim3d([-4.0, 4.0])
ax.set_ylabel('Y')
ax.set_zlim3d([-4.0, 4.0])
ax.set_zlabel('Z')
ax.set_title('3D Experiment')

# 4. plot surface
ax.plot_surface(xz, yz, zz, color='r')
ax.plot_surface(xc, yc, zc, color='lightyellow')
ax.plot_surface(xconst, yconst, zconst, color='navy')

ani = FuncAnimation(fig, update, 100, fargs=([xz,yz,zz],[xc,yc,zc], [xconst, yconst, zconst]),
                    interval=25, blit=False)

plt.show()