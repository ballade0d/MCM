import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
frequency = pd.read_excel('词频2.xlsx')

# 合并两个表
data = data.merge(frequency, on='Word')

# 将frequency最后的百分号去掉
data['frequency'] = data['frequency'].str[:-1].astype(float)
data['score'] = (0.5 * (1 * data['1 try'] + 2 * data['2 tries'] + 3 * data['3 tries'] + 4 * data['4 tries'] + 5 * data[
    '5 tries'] + 6 * data['6 tries']) + 0.5 * np.exp(-data['7 or more tries (X)'])) / 100
data['percentage'] = data['Number of  reported results'] / data['Number in hard mode']

# 选择需要聚类的特征
features = data[['score', 'frequency', 'percentage']]

# 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# KMeans聚类
kmeans = KMeans(n_clusters=3, init='k-means++')
data['difficulty'] = kmeans.fit_predict(features)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['difficulty'])

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('plot/kmeans.pdf')
plt.show()

# 计算轮廓系数（评估聚类效果）
score = silhouette_score(features, data['difficulty'])
print("Silhouette Coefficient: %0.3f" % score)

my_word = 'eerie'
my_frequency = 0.0002396
my_try = [0.61, 5.69, 24.75, 34.88, 23.11, 8.92, 1.38]
my_score = (0.5 * (1 * my_try[0] + 2 * my_try[1] + 3 * my_try[2] + 4 * my_try[3] + 5 * my_try[4] + 6 * my_try[5]) + 0.5 * my_try[6]) / 100
print(my_score)

# 预测
pred = kmeans.predict(scaler.transform([[np.mean(my_try), my_frequency, 1]]))

# 画所有单词按照score排列的图
data = data.sort_values(by='score', ascending=True)
plt.scatter(data['Word'], data['score'], s=5)
# 找到最近的score的位置
index = np.abs(data['score'] - my_score).idxmin()
plt.scatter(data['Word'][index], data['score'][index], c='red', s=50)
plt.title('Score')
# x轴不显示数值
plt.xticks([])
plt.savefig('plot/score.pdf')
plt.show()
