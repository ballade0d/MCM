import pandas as pd

athletes = pd.read_csv('data/summerOly_athletes.csv')
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

'''计算美国每年新晋运动员的数量'''
new_us_athletes = athletes[athletes['NOC'] == 'USA']
# 找到每位运动员首次参加比赛的年份
first_year = new_us_athletes.groupby('Name')['Year'].min().reset_index()
first_year.columns = ['Name', 'First_Year']
# 按首次参赛年份进行统计
new_us_athletes_per_year = first_year.groupby('First_Year')['Name'].count()

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = new_us_athletes_per_year.index.values.reshape(-1, 1)
y = new_us_athletes_per_year.values

# 过滤掉 1908 年以前的数据
X = X[4:]
y = y[4:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model on the Test Set
score = model.score(X_test, y_test)
print("R^2 Score on Test Set:", score)
# MSE
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Predict 2028
future_data = pd.DataFrame({'Year': [2028]})  # Host country in 2028
future_data_scaled = scaler.transform(future_data)  # Apply the same scaling
future_prediction = model.predict(future_data_scaled)
print("2028预测：", future_prediction)

# visualize
import matplotlib.pyplot as plt

plt.plot(new_us_athletes_per_year.index, new_us_athletes_per_year.values, label='Historical')
plt.plot([2028], future_prediction, 'ro', label='2028 Prediction')
plt.title('New Athletes from USA')
plt.legend()
plt.show()

# draw the prediction line
import numpy as np

X_all = np.concatenate((X, [[2028]]))
X_all_scaled = scaler.transform(X_all)
y_all = model.predict(X_all_scaled)

plt.plot(X_all, y_all, label='Prediction Line')
plt.scatter(new_us_athletes_per_year.index, new_us_athletes_per_year.values, label='Historical')
plt.plot([2028], future_prediction, 'ro', label='2028 Prediction')
plt.title('New Athletes from USA')
plt.legend()
plt.show()
