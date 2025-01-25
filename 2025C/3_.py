import matplotlib.pyplot as plt
import pandas as pd

medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

# 获取每个国家rank和year，组成一个新的dataframe
medal_counts = medal_counts[['NOC', 'Year', 'Rank']]

rank = {}

for country in medal_counts['NOC'].unique():
    # 获取每个国家的数据
    country_data = medal_counts[medal_counts['NOC'] == country]
    rank[country] = {}
    for year in country_data['Year'].unique():
        # 获取每个国家每年的数据
        year_data = country_data[country_data['Year'] == year]
        # 获取每个国家每年的rank
        rank[country][year] = year_data['Rank'].values[0]

# 统计每个国家参加奥运会的次数
appearances = {}
for country in rank:
    appearances[country] = len(rank[country])

# 画appearance和rank的散点图
plt.scatter(appearances.values(), [min(rank[country].values()) for country in rank])
plt.xlabel('Appearances')
plt.ylabel('Best Rank')
plt.title('Appearances vs Best Rank')
plt.show()
