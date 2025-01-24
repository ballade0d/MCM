import matplotlib.pyplot as plt
import pandas as pd

athletes = pd.read_csv('data/summerOly_athletes.csv')
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

# draw medal count trend of a country
usa = medal_counts[medal_counts['NOC'] == 'United States']
plt.plot(usa['Year'], usa['Gold'], label='Gold')
plt.plot(usa['Year'], usa['Silver'], label='Silver')
plt.plot(usa['Year'], usa['Bronze'], label='Bronze')
plt.legend()
plt.show()

'''计算2024年美国运动员的数量'''
us_athletes = athletes[athletes['Year'] == 2024]
print(us_athletes)
us_athletes = us_athletes[us_athletes['NOC'] == 'USA']
us_athletes_count = us_athletes['Name'].nunique()
print("2024年美国运动员的数量：")
print(us_athletes_count)

'''计算美国每年新晋运动员的数量'''
new_us_athletes = athletes[athletes['NOC'] == 'USA']
# 找到每位运动员首次参加比赛的年份
first_year = new_us_athletes.groupby('Name')['Year'].min().reset_index()
first_year.columns = ['Name', 'First_Year']
# 按首次参赛年份进行统计
new_us_athletes_per_year = first_year.groupby('First_Year')['Name'].count()
print("美国每年新晋运动员的数量：")
print(new_us_athletes_per_year)
# draw the trend
plt.plot(new_us_athletes_per_year.index, new_us_athletes_per_year.values)
plt.show()

'''获取2024年美国新晋运动员的获奖率'''
# 筛选出美国运动员
us_athletes = athletes[athletes['NOC'] == 'USA']
# 找到每位运动员首次参赛年份
first_year = us_athletes.groupby('Name')['Year'].min().reset_index()
first_year.columns = ['Name', 'First_Year']
# 标记运动员类型：新晋（2024年首次参赛）或老运动员
us_athletes = us_athletes.merge(first_year, on='Name', how='left')
us_athletes['Athlete_Type'] = us_athletes['First_Year'].apply(lambda x: 'New' if x == 2024 else 'Old')
# 筛选2024年的比赛数据
us_athletes_2024 = us_athletes[us_athletes['Year'] == 2024]
# 判断哪些运动员获奖
us_athletes_2024['Is_Winner'] = us_athletes_2024['Medal'] != 'No medal'
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

# 输出结果
print(f"2024年美国新晋运动员总数：{total_new_athletes}")
print(f"2024年美国新晋运动员的获奖人数：{winning_new_athletes}")
print(f"2024年美国新晋运动员的获奖率为：{winning_rate_new:.2%}")

print(f"2024年美国老运动员总数：{total_old_athletes}")
print(f"2024年美国老运动员的获奖人数：{winning_old_athletes}")
print(f"2024年美国老运动员的获奖率为：{winning_rate_old:.2%}")

'''计算所有运动员参加的比赛次数'''
# columns: Name, Year
# 可能有重复的运动员名字，一年可能参加多次比赛
athletes = athletes[['Name', 'Year']]
# 查看一个运动员有多少个不同的年份
unique_years_per_athlete = athletes.groupby('Name')['Year'].nunique()
unique_years_per_athlete = unique_years_per_athlete.value_counts().sort_index()
print(unique_years_per_athlete)
# find the max athlete

# 计算总运动员人数
total_athletes = unique_years_per_athlete.sum()

# 计算参加 1 次、2 次、3 次比赛的运动员数量
count_1 = unique_years_per_athlete.get(1, 0)  # 如果没有运动员只参加 1 次比赛，返回 0
count_2 = unique_years_per_athlete.get(2, 0)  # 如果没有运动员只参加 2 次比赛，返回 0
count_3 = unique_years_per_athlete.get(3, 0)  # 如果没有运动员只参加 3 次比赛，返回 0

# 计算概率
prob_1 = count_1 / total_athletes if total_athletes > 0 else 0
prob_2 = count_2 / total_athletes if total_athletes > 0 else 0
prob_3 = count_3 / total_athletes if total_athletes > 0 else 0

# 打印结果
print(f"参加 1 次比赛的概率：{prob_1:.2%}")
print(f"参加 2 次比赛的概率：{prob_2:.2%}")
print(f"参加 3 次比赛的概率：{prob_3:.2%}")
