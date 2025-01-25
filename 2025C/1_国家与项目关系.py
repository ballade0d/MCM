import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

medals = pd.read_csv("data/medals_percentage_per_sport.csv", index_col='NOC')

threshold = 0.8

# 筛选掉奖牌总数少于阈值的国家和项目
medals = medals.loc[
    medals.sum(axis=1) > threshold,  # 国家总奖牌数大于阈值
    medals.sum(axis=0) > threshold  # 项目总奖牌数大于阈值
]

# 绘制热力图
plt.figure(figsize=(10, 12))
sns.heatmap(medals, annot=False, fmt=".0f", cmap="YlGnBu", cbar=True)

# 添加标题和轴标签
plt.title("Heatmap of Medals by Country (NOC) and Sport", fontsize=16)
plt.xlabel("Sport", fontsize=12)
plt.ylabel("Country (NOC)", fontsize=12)

# 展示图表
plt.show()
