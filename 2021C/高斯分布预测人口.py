import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

data = pd.read_excel("2021MCMProblemC_DataSet.xlsx")

pos = np.column_stack((data['Longitude'], data['Latitude']))

print(pos)

# 绘制原始数据地图
plt.figure()
plt.scatter(data['Longitude'], data['Latitude'], c='r', marker='o')
plt.title("Original Geographic Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

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
print(density.min())
print(density.max())

colors = [(0, 0, 1, 0), (0, 0, 1, 1)]  # RGBA格式，从不透明的蓝色到完全透明
cmap_name = 'blue_transparent'
blue_transparent = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

# Plot the KDE result as a density heatmap
plt.figure()
plt.imshow(
    density, origin='lower', aspect='auto',
    extent=(x_min, x_max, y_min, y_max),
    cmap=blue_transparent, alpha=0.9,
)
plt.colorbar(label='Density')
# plt.scatter(data['Longitude'], data['Latitude'], c='r', marker='o', s=10, alpha=0.5)
plt.title("Geographic Data with KDE Density Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 导出heatmap，隐藏坐标，设置背景为透明
plt.figure()
plt.imshow(
    density, origin='lower', aspect='auto',
    extent=(x_min, x_max, y_min, y_max),
    cmap=blue_transparent, alpha=0.9,
)
plt.axis('off')
plt.savefig("heatmap.svg", transparent=True)
