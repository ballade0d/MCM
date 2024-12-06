import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_grid(ax, rows, cols, facecolor, alpha):
    for row in range(rows):
        for col in range(cols):
            square = patches.Circle((col, row), 0.1, facecolor=facecolor, alpha=alpha)
            ax.add_patch(square)

# 设置网格的行数和列数
rows = 50
cols = 65
facecolor = '#007BFF'  # 更深的蓝色
alpha = 1  # 半透明

# 创建画布和坐标轴，设置画布背景透明
fig, ax = plt.subplots()
fig.patch.set_facecolor('none')  # 设置绘图区背景透明
fig.patch.set_alpha(0.0)  # 设置绘图区透明度
ax.patch.set_alpha(0.0)  # 设置坐标轴区域透明度

draw_grid(ax, rows, cols, facecolor, alpha)

# 设置坐标轴的显示范围和比例
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')

# 隐藏坐标轴
ax.axis('off')

# 显示图形
plt.show()

# 如果需要保存图像，确保背景透明
fig.savefig('grid.svg', transparent=True)