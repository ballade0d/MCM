import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Bidirectional, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

athletes = pd.read_csv('data/summerOly_athletes.csv')
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

'''计算美国每年新晋运动员的数量'''
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
print("2028预测：", future_prediction)