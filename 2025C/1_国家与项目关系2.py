import matplotlib.pyplot as plt
import pandas as pd

t = pd.read_csv("data/占50%以上的项目.csv", index_col='NOC')
# 读取数据
medals = pd.read_csv("data/medals_per_sport.csv", index_col='NOC')
# medals仅保留t中index包含的国家
medals = medals[medals.index.isin(t.index)]
print(medals)

# 绘制堆叠柱状图
medals.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="tab20")

# 添加标题和轴标签
plt.title("Stacked Bar Chart of Medals by Sport and Country (NOC)", fontsize=16)
plt.xlabel("Sport", fontsize=12)
plt.ylabel("Number of Medals", fontsize=12)

# 添加图例
plt.legend(title="Country (NOC)", bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.show()
