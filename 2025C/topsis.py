import pandas as pd
import numpy as np

# 读取数据
athletes = pd.read_csv('data/summerOly_athletes.csv')

# 假设我们以运动员的金牌数、银牌数、铜牌数作为评价指标
# 首先，需要对每个运动员的奖牌数量进行统计
medal_counts = athletes.groupby('Name')['Medal'].value_counts().unstack(fill_value=0)

# 重命名列名，便于识别
medal_counts = medal_counts.rename(columns={'Gold': 'GoldMedals', 'Silver': 'SilverMedals', 'Bronze': 'BronzeMedals', 'No Medal': None})

# 如果有需要，添加其他指标

# 将缺失的指标列补充为0（如果某些运动员未获得某种奖牌）
for col in ['GoldMedals', 'SilverMedals', 'BronzeMedals']:
    if col not in medal_counts.columns:
        medal_counts[col] = 0

# 提取指标数据
data = medal_counts[['GoldMedals', 'SilverMedals', 'BronzeMedals']].values

# 标准化处理（归一化）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

# 指标权重设置（假设金牌权重最高）
weights = np.array([0.5, 0.3, 0.2])

# 计算加权标准化矩阵
data_weighted = data_norm * weights

# 正理想解和负理想解
ideal_best = np.max(data_weighted, axis=0)
ideal_worst = np.min(data_weighted, axis=0)

# 计算与正理想解和负理想解的距离
distance_to_best = np.sqrt(np.sum((data_weighted - ideal_best) ** 2, axis=1))
distance_to_worst = np.sqrt(np.sum((data_weighted - ideal_worst) ** 2, axis=1))

# 计算综合得分
scores = distance_to_worst / (distance_to_best + distance_to_worst)

# 将得分添加回数据框
medal_counts['TOPSIS_Score'] = scores

# 排序
medal_counts = medal_counts.sort_values(by='TOPSIS_Score', ascending=False)

# 查看结果
print(medal_counts)

# 输出结果到文件
medal_counts.to_csv('topsis.csv')
