import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 假设您的数据文件位于 'data/summerOly_athletes.csv'
athletes = pd.read_csv('data/summerOly_athletes.csv')

# 只保留获得奖牌的运动员
medalists = athletes[athletes['Medal'] != 'No medal']
# 查看国家代码（NOC）的列表
countries = medalists['NOC'].unique()

# 选择几个主要国家（您可以根据需要修改）
selected_countries = ['USA', 'CHN', 'RUS', 'GBR', 'GER', 'FRA']

# 过滤出选定国家的数据
country_medals = medalists[medalists['NOC'].isin(selected_countries)]

# 按国家和体育项目统计奖牌数量
medal_counts = country_medals.groupby(['NOC', 'Sport']).size().reset_index(name='Medal_Count')

# 创建透视表，以 'Sport' 为索引，以 'NOC' 为列，值为 'Medal_Count'
pivot_table = medal_counts.pivot(index='Sport', columns='NOC', values='Medal_Count')

# 将缺失值填充为 0
pivot_table = pivot_table.fillna(0)

# 按总奖牌数对体育项目排序
pivot_table['Total'] = pivot_table.sum(axis=1)
pivot_table = pivot_table.sort_values(by='Total', ascending=False)
pivot_table = pivot_table.drop(columns='Total')

# 选择前10个体育项目
top_sports = pivot_table.head(10)

# 设置图形大小
plt.figure(figsize=(12, 8))

# 绘制堆叠柱状图
top_sports.plot(kind='bar', stacked=True)

# 添加标题和标签
plt.title('Top 10 Sports by Medal Distribution')
plt.xlabel('Sport')
plt.ylabel('Medal Count')

# 设置图例位置
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()
plt.show()

# 重置索引，准备用于 Seaborn
data_for_sns = top_sports.reset_index().melt(id_vars='Sport', var_name='NOC', value_name='Medal_Count')

# 绘图
plt.figure(figsize=(12, 8))
sns.barplot(data=data_for_sns, x='Sport', y='Medal_Count', hue='NOC')

# 添加标题和标签
plt.title('Country-wise Medal Distribution for Top 10 Sports')
plt.xlabel('Sport')
plt.ylabel('Medal Count')

# 设置 x 轴标签旋转
plt.xticks(rotation=45)

# 设置图例位置
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()
plt.show()