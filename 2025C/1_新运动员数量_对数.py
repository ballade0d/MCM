import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 数据读取与处理
athletes = pd.read_csv('data/summerOly_athletes.csv')
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

# 计算美国每年新晋运动员的数量
new_us_athletes = athletes[athletes['NOC'] == 'USA']
# 找到每位运动员首次参加比赛的年份
first_year = new_us_athletes.groupby('Name')['Year'].min().reset_index()
first_year.columns = ['Name', 'First_Year']
# 按首次参赛年份进行统计
new_us_athletes_per_year = first_year.groupby('First_Year')['Name'].count()

# 准备数据
X = new_us_athletes_per_year.index.values.reshape(-1, 1)  # 年份
y = new_us_athletes_per_year.values                       # 新晋运动员数量

# 过滤掉 1908 年以前的数据
X = X[4:]
y = y[4:]

# 对目标值 y 进行对数变换
y_log = np.log(y)

# 分割数据集
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.1, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练对数回归模型（线性回归用于对数值）
model = LinearRegression()
model.fit(X_train_scaled, y_train_log)

# 模型评估
y_test_pred_log = model.predict(X_test_scaled)               # 预测对数值
y_test_pred = np.exp(y_test_pred_log)                        # 反对数转换为原始值
score = model.score(X_test_scaled, y_test_log)               # 对数空间的 R^2
mse = mean_squared_error(np.exp(y_test_log), y_test_pred)    # 原始空间的 MSE

print("R^2 Score on Test Set (Log Space):", score)
print("Mean Squared Error on Test Set (Original Space):", mse)

# 预测2028年的新晋运动员数量
future_data = pd.DataFrame({'Year': [2028]})
future_data_scaled = scaler.transform(future_data)
future_prediction_log = model.predict(future_data_scaled)    # 预测对数值
future_prediction = np.exp(future_prediction_log)            # 反对数转换为原始值
print("2028预测新晋运动员数量：", future_prediction)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(new_us_athletes_per_year.index, new_us_athletes_per_year.values, label='Historical Data', marker='o')
plt.scatter([2028], future_prediction, color='red', label='2028 Prediction', zorder=5)
plt.title('New Athletes from USA (Log Regression)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('New Athletes', fontsize=12)
plt.legend()
plt.show()

# 绘制对数回归预测线
X_all = np.concatenate((X, [[2028]]))
X_all_scaled = scaler.transform(X_all)

y_all_log = model.predict(X_all_scaled)

plt.figure(figsize=(10, 6))
plt.plot(X_all_scaled, y_all_log, label='Prediction Line')
plt.scatter(X_train_scaled, y_train_log, color='black', label='Training Data')
plt.title('New Athletes from USA (Log Regression)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('New Athletes', fontsize=12)
plt.legend()
plt.show()
