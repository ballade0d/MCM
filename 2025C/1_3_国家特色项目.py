import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据文件
athletes = pd.read_csv('data/summerOly_athletes.csv')

# 只保留获得奖牌的记录
medalists = athletes[athletes['Medal'] != 'No medal']

# 统计每个国家的总奖牌数
country_total_medals = medalists.groupby('NOC').size().reset_index(name='Total_Medals')

# 按照总奖牌数降序排序
country_total_medals = country_total_medals.sort_values(by='Total_Medals', ascending=False)

# 选择总奖牌数排名前 10 的国家
top_countries = country_total_medals['NOC'].head(10).tolist()

# 打印选定的国家列表
print("总奖牌数排名前 10 的国家：", top_countries)

# 过滤出选定国家的数据
country_medals = medalists[medalists['NOC'].isin(top_countries)]

# 统计各国家在 "Athletics" 项目上的奖牌数
athletics_medals = medalists[medalists['Sport'] == 'Athletics']
country_athletics_medals = athletics_medals.groupby('NOC').size().reset_index(name='Athletics_Medals')

# 筛选在田径项目上获得至少 20 枚奖牌的国家
countries_with_20_athletics_medals = country_athletics_medals[country_athletics_medals['Athletics_Medals'] >= 20]['NOC'].tolist()

# 打印选定的国家列表
print("在田径项目上获得至少 20 枚奖牌的国家：", countries_with_20_athletics_medals)

# 过滤出选定国家的数据
country_medals = medalists[medalists['NOC'].isin(countries_with_20_athletics_medals)]

# 统计每个国家在各体育项目上获得的奖牌数量
medal_counts = country_medals.groupby(['NOC', 'Sport']).size().reset_index(name='Medal_Count')

# 计算每个国家的奖牌总数
total_medals = medal_counts.groupby('NOC')['Medal_Count'].sum().reset_index(name='Total_Medals')

# 将总奖牌数合并回原数据框
medal_counts = pd.merge(medal_counts, total_medals, on='NOC')

# 计算每个国家在各项目上的奖牌占比
medal_counts['Medal_Percentage'] = medal_counts['Medal_Count'] / medal_counts['Total_Medals'] * 100

# 创建透视表，国家为列，体育项目为行，值为奖牌占比
heatmap_data = medal_counts.pivot_table(index='Sport', columns='NOC', values='Medal_Percentage', fill_value=0)

# 为了使图表更加清晰，可以选择对多数国家重要的体育项目
# 选择平均奖牌占比最高的前 20 个体育项目
top_sports = heatmap_data.mean(axis=1).sort_values(ascending=False).head(20).index
heatmap_data = heatmap_data.loc[top_sports]

# 绘制热力图
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu')
# x轴旋转45度
plt.xticks(rotation=45)
# 添加标题
plt.title('Heatmap of Medal Distribution by Sport and Country (Top 20 Sports)')

# 显示图表
plt.tight_layout()
plt.savefig('plot/Heatmap_of_Medal_Distribution_by_Sport_and_Country1.pdf')
plt.show()

# 绘制聚类热力图
sns.clustermap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu', figsize=(14, 12))

# 添加标题
plt.suptitle('Heatmap of Medal Distribution by Sport and Country (Top 20 Sports)', y=1.05)

plt.savefig('plot/Heatmap_of_Medal_Distribution_by_Sport_and_Country.pdf')
# 显示图表
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 转置数据，使之适合 PCA（国家为样本，体育项目为特征）
pca_data = heatmap_data.T

# 标准化数据
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

# 应用 PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(pca_data_scaled)

# 创建数据框保存结果
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['NOC'] = pca_data.index

# 绘制散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='NOC', s=100)

# 添加标题和标签
plt.title('PCA of Medal Distribution by Sport and Country')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.savefig('plot/PCA_of_Medal_Distribution_by_Sport_and_Country.pdf')
# 显示图表
plt.show()

# 定义一个函数，找到每个国家奖牌占比最高的前三个体育项目
def get_top_sports(df, top_n=3):
    top_sports_per_country = pd.DataFrame()
    for country in df['NOC'].unique():
        country_data = df[df['NOC'] == country]
        top_sports = country_data.sort_values(by='Medal_Percentage', ascending=False).head(top_n)
        top_sports_per_country = pd.concat([top_sports_per_country, top_sports], ignore_index=True)
    return top_sports_per_country

# 获取每个国家的特色项目
top_sports_per_country = get_top_sports(medal_counts, top_n=3)

# 查看结果
print(top_sports_per_country)

# 绘制各国特色项目的柱状图
plt.figure(figsize=(10, 6))
sns.barplot(data=top_sports_per_country, x='NOC', y='Medal_Percentage', hue='Sport')

# 添加标题和标签
plt.title('Top 3 Sports by Medal Percentage for Each Country')
plt.xlabel('Country')
plt.ylabel('Medal Percentage')
# x轴旋转45度
plt.xticks(rotation=45)

# 设置图例位置
plt.legend(title='Sport', bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()
plt.savefig('plot/Top_3_Sports_by_Medal_Percentage_for_Each_Country.pdf')
plt.show()
