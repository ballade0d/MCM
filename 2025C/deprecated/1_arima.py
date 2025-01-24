import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

medal_counts = pd.read_csv('../data/summerOly_medal_counts.csv')
hosts = pd.read_csv('../data/summerOly_hosts.csv')

# calculate weighted historical medal counts
# USA only
medal_counts = medal_counts[medal_counts['NOC'] == 'United States']
# check host
medal_counts['host'] = 0
# reset index
medal_counts = medal_counts.reset_index(drop=True)
for i in range(len(medal_counts)):
    year = medal_counts['Year'][i]
    host = hosts[hosts['Year'] == year]['Host'].values[0]
    country = medal_counts['NOC'][i]
    if host == country:
        medal_counts.loc[i, 'host'] = 1

# use arima model to predict total medal counts
from statsmodels.tsa.arima.model import ARIMA

# 检查平稳性
result = adfuller(medal_counts['Total'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# 绘制 ACF 和 PACF 图
plot_acf(medal_counts['Total'], lags=20, title="ACF")
plot_pacf(medal_counts['Total'], lags=13, title="PACF")
plt.show()

# 由 ACF 和 PACF 图确定 p 和 q 值
p = 1
d = 0
q = 1

# 创建并拟合 ARIMA 模型

# use train_test_split to split the data
from sklearn.model_selection import train_test_split

X = medal_counts['Year']
y = medal_counts['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = ARIMA(y_train, order=(p, d, q))
model_fit = model.fit()

# 预测
y_pred = model_fit.forecast(steps=len(y_test))
# 评估
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# 预测 2028
future_prediction = model_fit.forecast(steps=len(y_test) + 1)
print("2028预测：", future_prediction)

# visualize the prediction
plt.plot(medal_counts['Year'], medal_counts['Total'], label='Historical')
plt.plot(medal_counts['Year'][-len(y_test):], y_pred, label='Prediction')
plt.plot([2028], future_prediction[27], 'ro', label='2028 Prediction')
plt.legend()
plt.show()
