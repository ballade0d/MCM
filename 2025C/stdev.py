import json

import pandas as pd

countries = ['Kenya', 'Turkey', 'Sweden', 'Canada', 'Great Britain', 'Romania', 'Australia', 'Greece', 'Thailand',
             'Spain', 'Iran', 'Belgium', 'Switzerland', 'Italy', 'Brazil', 'Denmark', 'Jamaica', 'Netherlands',
             'Morocco', 'China', 'Indonesia', 'Bulgaria', 'Hungary', 'South Korea', 'Argentina', 'Poland', 'Norway',
             'Mexico', 'Japan', 'United States', 'France', 'New Zealand']

medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

# 获取每个国家rank和year，组成一个新的dataframe
medal_counts = medal_counts[['NOC', 'Year', 'Rank']]

rank = {}

for country in countries:
    # 获取每个国家的数据
    country_data = medal_counts[medal_counts['NOC'] == country]
    rank[country] = {}
    for year in country_data['Year'].unique():
        # 获取每个国家每年的数据
        year_data = country_data[country_data['Year'] == year]
        # 获取每个国家每年的rank
        rank[country][year] = year_data['Rank'].values[0]

# get standard deviation of rank
std = {}
for country in rank:
    std[country] = pd.Series(rank[country]).std()
# drop nan
std = {k: v for k, v in std.items() if pd.notna(v)}

# sort by standard deviation
std = dict(sorted(std.items(), key=lambda x: x[1], reverse=True))

print(json.dumps(std, indent=2))
