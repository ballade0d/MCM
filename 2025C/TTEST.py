import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

medal_counts = pd.read_csv("data/summerOly_medal_counts.csv")
# cols: Year, Host
hosts = pd.read_csv("data/summerOly_hosts.csv", index_col='Year')

data = pd.DataFrame(columns=['Country', 'Medal Count', 'Host'])

for country in hosts['Host'].unique():
    if country == 'Soviet Union' or country == 'West Germany':
        continue
    if country == 'United Kingdom':
        country = 'Great Britain'
    host_count = 0.0
    host_num = 0.0
    non_host_count = 0
    non_host_num = 0

    host = []
    non_host = []
    medal_count = medal_counts[medal_counts['NOC'] == country]
    for row in medal_count.iterrows():
        # check if the country is host
        if hosts.loc[row[1]['Year']]['Host'] == country or (country == 'Great Britain' and hosts.loc[row[1]['Year']]['Host'] == 'United Kingdom'):
            host_count += row[1]['Total']
            host_num += 1
            host += [row[1]['Total']]
        else:
            non_host_count += row[1]['Total']
            non_host_num += 1
            non_host += [row[1]['Total']]
    if host_num != 0:
        host_count /= host_num
    if non_host_num != 0:
        non_host_count /= non_host_num

    # ttest
    t_stat, p_val = stats.ttest_ind(host, non_host)
    print(f'{country}: t-statistic = {t_stat}, p-value = {p_val}')


    data = pd.concat([data, pd.DataFrame([[country, host_count, True]], columns=['Country', 'Medal Count', 'Host'])])
    data = pd.concat([data, pd.DataFrame([[country, non_host_count, False]], columns=['Country', 'Medal Count', 'Host'])])

print(data)
# 设置图表大小
plt.figure(figsize=(16, 5))

countries = data['Country'].unique()
host_data = data[data['Host'] == True]['Medal Count'].values
non_host_data = data[data['Host'] == False]['Medal Count'].values
# 计算条形图的位置
positions = np.arange(len(countries)) * 2  # 两倍间隔确保空间足够分辨两个条
bar_width = 0.8

# 绘图
plt.bar(positions - bar_width/2, host_data, width=bar_width, color='r', label='Host', alpha=0.6)
plt.bar(positions + bar_width/2, non_host_data, width=bar_width, color='b', label='Non-Host', alpha=0.6)

# 设置 x 轴标签
plt.xticks(positions, countries, rotation=45, fontsize=12)

# 添加图表元素
plt.ylabel('Average Medal Count', fontsize=14)
plt.title('Average Medal Count of Host and Non-Host Countries', fontsize=16)
plt.legend()

# 优化布局并保存图像
plt.tight_layout()
plt.savefig('plot/Host_Non-Host_Medal_Count.pdf')

# 显示图表
plt.show()