import random

import matplotlib
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 第一步：确定人口密度
data = pd.read_excel("2021MCMProblemC_DataSet.xlsx")

pos = np.column_stack((data['Longitude'], data['Latitude']))

# KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
kde.fit(pos)

# 生成网格点
# -125.4441467 -116.0945543
# 45.0827575 49.9539355
x_min, x_max = -125.4441467, -116.0945543
y_min, y_max = 45.0827575, 49.9539355
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 130), np.linspace(y_min, y_max, 100))
grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# 计算概率密度
log_density = kde.score_samples(grid_positions)
density = np.exp(log_density).reshape(x_grid.shape)
density = np.log1p(density)
density = np.sqrt(density)
S = 1 - density


def softmax(S_adj, S_cur, C_adj):
    # Calculate numerator
    numerator = np.exp(S_adj - S_cur)

    # Calculate denominator
    denominator = np.sum(np.exp(t - S_cur) for t in C_adj)

    # Compute softmax
    softmax_value = numerator / denominator
    return softmax_value


# 地图大小
width = 130
height = 100

# 创建网格
grid = np.zeros((height, width), dtype=float)  # 物种分布

# 初始物种位置
initial_positions = [
    (30, 50)
]
for pos in initial_positions:
    grid[pos[0], pos[1]] = 1000

neighbors = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 0), (0, 1),
             (1, -1), (1, 0), (1, 1)]

increase_month = [2, 3, 4, 5, 6, 7, 8, 9, 10]
current_month = 9


def update_species(frameNum, img, species):
    global current_month

    for i in range(height):
        for j in range(width):
            if species[i, j] == 0:
                continue
            '''
            old = species[i, j]
            multiplier = 1.2 if current_month in increase_month else 0.995
            multiplier *= (1 - species[i, j] / 10000)
            multiplier *= S[i, j]
            species[i, j] *= multiplier
            if old > 50 > species[i, j]:
                species[i, j] = 51
            '''


            old = species[i, j]
            r = 0.15 if current_month in increase_month else -0.005
            species[i, j] = old + old * r * (1 - old / 5000)
            species[i, j] *= S[i, j]

            if old > 50 > species[i, j]:
                species[i, j] = 51


    new_species = np.copy(species)

    for i in range(height):
        for j in range(width):
            if species[i, j] != 0:
                neighbors_sum = []
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    neighbors_sum.append(S[ni, nj])

                total_move = 0

                for di, dj in neighbors:
                    if (di, dj) == (0, 0):
                        continue
                    ni, nj = i + di, j + dj
                    # Check boundaries
                    if 0 <= ni < height and 0 <= nj < width:
                        T = softmax(S[ni, nj], S[i, j], neighbors_sum) * (
                            0.3 if current_month in increase_month else 0.1)
                        if random.random() < T:
                            new_species[ni, nj] += new_species[i, j] * T
                            total_move += new_species[i, j] * T
                new_species[i, j] = new_species[i, j] - total_move

    new_species[new_species < 1e-5] = 0
    # Update the image data
    img.set_data(new_species)
    species[:] = new_species

    if frameNum == 30:
        current_month += 1
        print("Month passed")
    if current_month == 13:
        current_month = 1
        print("Year passed")
    return new_species


# 设置动画
fig = plt.figure(frameon=False)

img1 = plt.imshow(
    density, origin='lower', aspect='auto',
    cmap='Blues',
)
img2 = plt.imshow(
    grid, interpolation='bilinear', cmap='Reds', alpha=0.5
)
ani = animation.FuncAnimation(fig, update_species, fargs=(img2, grid,),
                              frames=31,
                              interval=10)

# 显示动画
plt.show()
