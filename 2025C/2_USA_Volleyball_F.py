import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
athletes = pd.read_csv("data/summerOly_athletes.csv")

# 筛选条件
country = 'USA'
sport = 'Volleyball'
sex = 'F'
years = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

# 筛选数据
data = athletes[
    (athletes['NOC'] == country) &
    (athletes['Sport'] == sport) &
    (athletes['Sex'] == sex) &
    (athletes['Year'].isin(years))
]

# 映射奖牌到对应分值（gold=3, silver=2, bronze=1, no medal=0）
medal_points = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
data['Points'] = data['Medal'].map(medal_points).fillna(0)  # 无奖牌的赋值为0

# 计算每一年最高奖牌（只取该年份的最高分值）
yearly_points = data.groupby('Year')['Points'].max()

# 确保每一年都有数据（即使某一年没有任何奖牌）
yearly_points = yearly_points.reindex(years, fill_value=0)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(yearly_points.index, yearly_points.values, marker='o', linestyle='-', color='b', label='Highest Medal Points')

# 在特定年份绘制标注线和文字
plt.axvline(x=2005, color='r', linestyle='--')
plt.text(2005.5, 1, 'Lang Ping Joined USA Volleyball', fontsize=12, color='r')

plt.axvline(x=2008, color='r', linestyle='--')
plt.text(2008.5, 2, 'Lang Ping Left USA Volleyball', fontsize=12, color='r')

plt.axvline(x=2013, color='r', linestyle='--')
plt.text(2013.5, 3, 'Karch Kiraly Joined USA Volleyball', fontsize=12, color='r')

# 图表标题和轴标签
plt.title(f'Medal Points Per Year ({country} - {sport} - {sex})', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Medal', fontsize=14)
plt.xticks(years, fontsize=12)
plt.yticks([0, 1, 2, 3], ['No Medal', 'Bronze', 'Silver', 'Gold'], fontsize=12)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()