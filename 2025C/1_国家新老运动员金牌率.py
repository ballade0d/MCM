import pandas as pd
from matplotlib import pyplot as plt

countries = ['USA']
years = [2024, 2020, 2016, 2012, 2008, 2002, 1996, 1992, 1988, 1984, 1980, 1976, 1972, 1968, 1964, 1960, 1956, 1952, 1948, 1944, 1940, 1936, 1932, 1928, 1924, 1920, 1916, 1912, 1908]

for country in countries:
    '''获取2024年新晋运动员的获奖率'''
    # 筛选出美国运动员
    athletes = pd.read_csv('data/summerOly_athletes.csv')

    new_rate = {}
    old_rate = {}
    for year in years:
        us_athletes = athletes[athletes['NOC'] == country]
        # 找到每位运动员首次参赛年份
        first_year = us_athletes.groupby('Name')['Year'].min().reset_index()
        first_year.columns = ['Name', 'First_Year']
        # 标记运动员类型：新晋（首次参赛）或老运动员
        us_athletes = us_athletes.merge(first_year, on='Name', how='left')
        us_athletes['Athlete_Type'] = us_athletes['First_Year'].apply(lambda x: 'New' if x == year else 'Old')
        # 筛选2024年的比赛数据
        us_athletes_2024 = us_athletes[us_athletes['Year'] == year]
        if len(us_athletes_2024) == 0:
            continue
        # 判断哪些运动员获奖
        us_athletes_2024['Is_Winner'] = us_athletes_2024['Medal'] == 'Gold'
        # 计算新晋运动员的获奖率
        new_athletes_2024 = us_athletes_2024[us_athletes_2024['Athlete_Type'] == 'New']
        total_new_athletes = new_athletes_2024['Name'].nunique()  # 新晋运动员总数
        winning_new_athletes = new_athletes_2024[new_athletes_2024['Is_Winner']]['Name'].nunique()  # 获奖新晋运动员数
        winning_rate_new = winning_new_athletes / total_new_athletes if total_new_athletes > 0 else 0
        # 计算老运动员的获奖率
        old_athletes_2024 = us_athletes_2024[us_athletes_2024['Athlete_Type'] == 'Old']
        total_old_athletes = old_athletes_2024['Name'].nunique()  # 老运动员总数
        winning_old_athletes = old_athletes_2024[old_athletes_2024['Is_Winner']]['Name'].nunique()  # 获奖老运动员数
        winning_rate_old = winning_old_athletes / total_old_athletes if total_old_athletes > 0 else 0

        new_rate[year] = winning_rate_new
        old_rate[year] = winning_rate_old
    # plot
    plt.plot(list(new_rate.keys()), list(new_rate.values()), label='New Athletes')
    plt.plot(list(old_rate.keys()), list(old_rate.values()), label='Old Athletes')
    plt.xlabel('Year')
    plt.ylabel('Gold Rate')
    plt.title(f'{country} Gold Rate')
    plt.legend()
    plt.show()
