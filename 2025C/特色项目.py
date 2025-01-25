import pandas as pd

# 假设 athletes 数据已加载
athletes = pd.read_csv('data/summerOly_athletes.csv')

# 过滤掉 "No medal"，保留有奖牌的数据
athletes = athletes[athletes['Medal'] != 'No medal']

# 按国家、年份、项目统计奖牌数
country_year_sport_medals = (
    athletes.groupby(['NOC', 'Year', 'Sport', 'Medal'])
    .size()
    .unstack(fill_value=0)  # 将 Medal 的类别展开（Gold, Silver, Bronze）
    .reset_index()
)

# 计算每个项目的总奖牌数（Gold + Silver + Bronze）
country_year_sport_medals['Total Medals'] = country_year_sport_medals[['Gold', 'Silver', 'Bronze']].sum(axis=1)

# 按国家、年份统计总奖牌数
country_year_totals = (
    country_year_sport_medals.groupby(['NOC', 'Year'])['Total Medals']
    .sum()
    .reset_index()
    .rename(columns={'Total Medals': 'Year Total Medals'})
)

# 将每年总奖牌数并入项目数据
country_year_sport_medals = pd.merge(
    country_year_sport_medals,
    country_year_totals,
    on=['NOC', 'Year']
)

# 计算每个项目的奖牌占比
country_year_sport_medals['Medal Share (%)'] = (
    country_year_sport_medals['Total Medals'] / country_year_sport_medals['Year Total Medals'] * 100
)

# 筛选出奖牌占比超过 20% 的项目
top_sport_medals = country_year_sport_medals[country_year_sport_medals['Medal Share (%)'] > 50]

# 输出结果
print(top_sport_medals)

# 保存结果
top_sport_medals.to_csv('占50%以上的项目.csv', index=False)