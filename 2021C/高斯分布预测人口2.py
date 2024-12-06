import matplotlib

matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

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
density = 1 - density
print(density.min())
print(density.max())

# 创建图形和3D轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图
surface = ax.plot_surface(x_grid, y_grid, density, cmap='viridis')

# 添加颜色条
fig.colorbar(surface)

# 设置标签
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Value')

# 显示图形
plt.show()
