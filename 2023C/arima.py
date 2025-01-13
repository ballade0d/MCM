import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

'''
1. 检查平稳性 (确定d值)
2. 绘制 ACF 和 PACF 图 (确定p和q值)
3. 使用p, d, q拟合ARIMA模型

d: 差分次数，用于将非平稳序列转换为平稳序列。
p: 自回归 (AR) 项数，表示时间序列当前值与过去 p 个值的线性关系。
q: 移动平均 (MA) 项数，表示时间序列当前值与过去 q 个预测误差项的关系。
规则：
如果 ACF 在某个滞后点之后快速衰减（接近零），而 PACF 缓慢衰减：
    说明是 AR(p) 模型，p 为 PACF 图中截断点的滞后阶数。q 可以为 0。
    
如果 PACF 在某个滞后点之后快速衰减，而 ACF 缓慢衰减：
    说明是 MA(q) 模型，q 为 ACF 图中截断点的滞后阶数。p 可以为 0。
    
如果 ACF 和 PACF 都是缓慢衰减：
    说明需要差分，或者可能是 ARMA（混合模型）。
    
注：适用于线性时间序列，不适用于非线性时间序列。所以效果不好。
'''

# 读取数据
data = pd.read_excel('Problem_C_Data_Wordle.xlsx')

# 打印列名确认数据结构
print(data.columns)

# 提取日期和报告结果数
date = data['Date']
contest_number = data['Number of  reported results']

# 设置日期为索引（确保数据按日期排序）
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# 检查平稳性
result = adfuller(contest_number)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    print("数据非平稳，需要差分处理")
else:
    print("数据平稳，不需要差分")

# 绘制 ACF 和 PACF 图
plot_acf(data['Number of  reported results'], lags=30, title="ACF")
plot_pacf(data['Number of  reported results'], lags=30, title="PACF")
plt.show()

# 由 ACF 和 PACF 图确定 p 和 q 值
p = 2
d = 0
q = 0

# 创建并拟合 ARIMA 模型
model = ARIMA(data['Number of  reported results'], order=(p, d, q))
model_fit = model.fit()

# 模型总结
print(model_fit.summary())

# 预测未来值
future_steps = 120
forecast = model_fit.forecast(steps=future_steps)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Number of  reported results'], label='Historical Data')
plt.plot(pd.date_range(data.index[-1], periods=future_steps + 1, freq='D')[1:], forecast, label='Forecast',
         linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of Reported Results')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
