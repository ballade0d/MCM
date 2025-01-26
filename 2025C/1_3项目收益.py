import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
athletes = pd.read_csv('data/summerOly_athletes.csv')

# 选择一个国家
country = 'FRA'

# 提取该国家的运动员数据
athletes_country = athletes[athletes['NOC'] == country]

# 提取该国家的奖牌获得者数据
medalists_country = athletes_country[athletes_country['Medal'] != 'No medal']

# 去除团队项目的重复计数
medal_events_country = medalists_country.drop_duplicates(subset=['Year', 'Sport', 'Event', 'Medal', 'Sex'])

# 计算每个项目每年的奖牌数
medal_counts_country = medal_events_country.groupby(['Sport', 'Year']).size().reset_index(name='Medal_Count')

# 创建透视表
medal_counts_pivot = medal_counts_country.pivot(index='Year', columns='Sport', values='Medal_Count').fillna(0)

# 计算平均收益和风险
returns = medal_counts_pivot.mean()
risks = medal_counts_pivot.std()

# 创建包含收益和风险的数据框
portfolio_data = pd.DataFrame({'Sport': returns.index, 'Return': returns.values, 'Risk': risks.values})

# 计算夏普比率
portfolio_data['Sharpe_Ratio'] = portfolio_data['Return'] / portfolio_data['Risk']
portfolio_data.replace([np.inf, -np.inf], np.nan, inplace=True)
portfolio_data.dropna(subset=['Sharpe_Ratio'], inplace=True)

# 选择夏普比率最高的前几个项目
top_n = 10
top_projects = portfolio_data.nlargest(top_n, 'Sharpe_Ratio')

# 绘制风险-收益散点图
plt.figure(figsize=(12, 8))
scatter = plt.scatter(portfolio_data['Risk'], portfolio_data['Return'], s=50,
                      c=portfolio_data['Sharpe_Ratio'], cmap='viridis')

# 只标注夏普比率最高的项目
for i in top_projects.index:
    plt.text(portfolio_data['Risk'][i], portfolio_data['Return'][i],
             portfolio_data['Sport'][i], fontsize=12)

plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return (Average Medal Count)')
plt.title(f'{country} Risk-Return Scatter Plot of Sports')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.savefig('plot/risk_return_scatter.pdf')
plt.show()

# 准备投资组合数据
returns_data = medal_counts_pivot
covariance_matrix = returns_data.cov()
expected_returns = returns_data.mean()

# 蒙特卡罗模拟
num_ports = 5000
num_assets = len(returns_data.columns)
results = np.zeros((4, num_ports))
weights_record = []

for i in range(num_ports):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = sharpe_ratio

results_frame = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe_Ratio', 'Dummy'])
results_frame['Weights'] = weights_record

# 找到最优组合
max_sharpe_idx = results_frame['Sharpe_Ratio'].idxmax()
max_sharpe_portfolio = results_frame.loc[max_sharpe_idx]

print("最优组合（夏普比率最高）：")
print("预期收益：", max_sharpe_portfolio['Return'])
print("预期风险：", max_sharpe_portfolio['Volatility'])
print("夏普比率：", max_sharpe_portfolio['Sharpe_Ratio'])
print("资产权重：")
for i, sport in enumerate(returns_data.columns):
    print(f"{sport}: {max_sharpe_portfolio['Weights'][i]:.4f}")

# 绘制有效前沿
plt.figure(figsize=(12, 8))
plt.scatter(results_frame['Volatility'], results_frame['Return'], c=results_frame['Sharpe_Ratio'], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], marker='*', color='r', s=500, label='Best Portfolio')
plt.xlabel('Expected Risk (Standard Deviation)')
plt.ylabel('Expected Return (Average Medal Count)')
plt.title(f'{country} Efficient Frontier of Portfolios (Random Simulation)')
plt.legend()
plt.grid(True)
plt.savefig('plot/efficient_frontier.pdf')
plt.show()

# 显示最优组合的项目权重
optimal_weights = pd.DataFrame({'Sport': returns_data.columns, 'Weight': max_sharpe_portfolio['Weights']})
optimal_weights = optimal_weights[optimal_weights['Weight'] > 0.01]
optimal_weights = optimal_weights.sort_values(by='Weight', ascending=False)
print("最优组合的项目及其权重：")
print(optimal_weights)