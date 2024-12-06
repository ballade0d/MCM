import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update(frameNum, img, grid, N):
    # 复制网格，因为我们需要基于原始数组进行更新
    newGrid = grid.copy()
    # 遍历每个细胞
    for i in range(N):
        for j in range(N):
            # 计算周围的活细胞总数
            total = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
                         grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
                         grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
                         grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N]))
            # 应用生命游戏的规则
            if grid[i, j] == 1:
                # 规则 1 和 3: 活细胞周围少于2个或多于3个活细胞则死亡
                if (total < 2) or (total > 3):
                    newGrid[i, j] = 0
            else:
                # 规则 2: 死细胞周围正好3个活细胞则变为活细胞
                if total == 3:
                    newGrid[i, j] = 1
    # 更新数据
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,


# 设置网格大小
N = 100

# 创建一个 N x N 的网格，初始状态随机
grid = np.random.choice([0, 1], N * N, p=[0.8, 0.2]).reshape(N, N)

# 设置动画
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='bilinear', cmap='viridis') # 颜色、样式
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N,),
                              frames=10,
                              interval=50)

# 显示动画
plt.show()
