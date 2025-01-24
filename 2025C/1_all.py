import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Bidirectional, Dropout
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

country = "USA"
years = [2024, 2020, 2016, 2012, 2008, 2002, 1996, 1992, 1988, 1984, 1980, 1976, 1972, 1968, 1964, 1960, 1956, 1952,
         1948, 1944, 1940, 1936, 1932, 1928, 1924, 1920, 1916, 1912, 1908]
target = 2028

'''获取2024年新晋运动员的获奖率'''
athletes = pd.read_csv('data/summerOly_athletes.csv')

new_rate = {}
old_rate = {}
new_count = {}
for year in years:
    country_athletes = athletes[athletes['NOC'] == country]
    # 找到每位运动员首次参赛年份
    first_year = country_athletes.groupby('Name')['Year'].min().reset_index()
    first_year.columns = ['Name', 'First_Year']
    # 标记运动员类型：新晋（首次参赛）或老运动员
    country_athletes = country_athletes.merge(first_year, on='Name', how='left')
    country_athletes['Athlete_Type'] = country_athletes['First_Year'].apply(lambda x: 'New' if x == year else 'Old')
    country_athletes_year = country_athletes[country_athletes['Year'] == year]
    if len(country_athletes_year) == 0:
        print(year)
        continue
    # 判断哪些运动员获奖
    country_athletes_year['Is_Winner'] = country_athletes_year['Medal'] != 'No medal'
    # 计算新晋运动员的获奖率
    new_athletes_year = country_athletes_year[country_athletes_year['Athlete_Type'] == 'New']
    total_new_athletes = new_athletes_year['Name'].nunique()  # 新晋运动员总数
    winning_new_athletes = new_athletes_year[new_athletes_year['Is_Winner']]['Name'].nunique()  # 获奖新晋运动员数
    winning_rate_new = winning_new_athletes / total_new_athletes if total_new_athletes > 0 else 0
    # 计算老运动员的获奖率
    old_athletes_year = country_athletes_year[country_athletes_year['Athlete_Type'] == 'Old']
    total_old_athletes = old_athletes_year['Name'].nunique()  # 老运动员总数
    winning_old_athletes = old_athletes_year[old_athletes_year['Is_Winner']]['Name'].nunique()  # 获奖老运动员数
    winning_rate_old = winning_old_athletes / total_old_athletes if total_old_athletes > 0 else 0

    new_rate[year] = winning_rate_new
    old_rate[year] = winning_rate_old
    new_count[year] = total_new_athletes
print(new_count)
'''预测target年新晋运动员的获奖率'''
print("预测新晋运动员的获奖率：")
X = list(new_rate.keys())
y = list(new_rate.values())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestRegressor()
model.fit([[x] for x in X_train], y_train)

# Evaluate the Model on the Test Set
score = model.score([[x] for x in X_test], y_test)
print("R^2 Score on Test Set:", score)
# MSE
y_pred = model.predict([[x] for x in X_test])

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Predict target
new_rate_prediction = model.predict([[target]])
print("target预测：", new_rate_prediction)

'''预测target年老运动员的获奖率'''
print("预测老运动员的获奖率：")
X = list(old_rate.keys())
y = list(old_rate.values())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestRegressor()
model.fit([[x] for x in X_train], y_train)

# Evaluate the Model on the Test Set
score = model.score([[x] for x in X_test], y_test)
print("R^2 Score on Test Set:", score)
# MSE
y_pred = model.predict([[x] for x in X_test])

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Predict target
old_rate_prediction = model.predict([[target]])
print("target预测：", old_rate_prediction)

'''预测target年新晋运动员的数量'''
print("预测新晋运动员的数量：")
new_us_athletes = athletes[athletes['NOC'] == 'USA']
# 找到每位运动员首次参加比赛的年份
first_year = new_us_athletes.groupby('Name')['Year'].min().reset_index()
first_year.columns = ['Name', 'First_Year']
# 按首次参赛年份进行统计
new_us_athletes_per_year = first_year.groupby('First_Year')['Name'].count()

X = new_us_athletes_per_year.values.reshape(-1, 1)


# 创建滞后序列
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, 0])  # 滞后窗口
        y.append(data[i, 0])  # 目标值
    return np.array(X), np.array(y)


# 数据归一化（缩放到 0-1 的范围，便于 LSTM 收敛）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(X)

lag = 3
X, y = create_lagged_data(scaled_values, lag)

# 将数据划分为训练集和测试集（80% 训练，20% 测试）
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(Bidirectional(LSTM(8, input_shape=(X_train.shape[1], 1), return_sequences=True)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, shuffle=False,
                    validation_data=(X_test, y_test), callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),  # 防止过拟合
        ReduceLROnPlateau(factor=0.8, patience=5)  # 动态调整学习率
    ])

# 使用测试集进行预测
predicted = model.predict(X_test)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plot/lstm_loss.pdf')
plt.show()

# 反归一化
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 绘制实际值和预测值对比（局部）
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual Values', color='blue')
plt.plot(predicted, label='Predicted Values', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Reported Results')
plt.legend()
plt.savefig('plot/lstm.pdf')
plt.show()

# 绘制实际值和预测值对比（全局）
plt.figure(figsize=(10, 6))
plt.plot(new_us_athletes_per_year.index, new_us_athletes_per_year.values, label='Historical')
plt.plot(new_us_athletes_per_year.index[train_size + lag:], predicted, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Values (Global)')
plt.xlabel('Time')
plt.ylabel('Number of New US Athletes')
plt.legend()
plt.savefig('plot/lstm_global.pdf')
plt.show()

# Predict the number of new athletes in 2028
future_data = X[-1:]
future_prediction = model.predict(future_data)
future_prediction = scaler.inverse_transform(future_prediction)
print("target预测：", future_prediction)

prev_new = new_count[years[0]]
print(future_prediction * new_rate_prediction + prev_new * old_rate_prediction * 0.196)
