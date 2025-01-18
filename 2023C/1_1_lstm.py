import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# 报告数
values = data['Number of  reported results'].values.reshape(-1, 1)

# 数据归一化（缩放到 0-1 的范围，便于 LSTM 收敛）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)


# 创建滞后序列（例如，使用过去 30 天的数据预测下一天）
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, 0])  # 滞后窗口
        y.append(data[i, 0])  # 目标值
    return np.array(X), np.array(y)


lag = 30  # 滞后窗口大小
X, y = create_lagged_data(scaled_values, lag)

# 将数据划分为训练集和测试集（80% 训练，20% 测试）
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 调整形状为 [samples, time steps, features]，以适应 LSTM 输入格式
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(Bidirectional(LSTM(8, input_shape=(X_train.shape[1], 1), return_sequences=True)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(4)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=False,
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
plt.plot(range(len(y_test_actual)), y_test_actual, label='Actual Values', color='blue')
plt.plot(range(len(predicted)), predicted, label='Predicted Values', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Reported Results')
plt.legend()
plt.savefig('plot/lstm_part.pdf')
plt.show()

# 绘制实际值和预测值对比（全局）
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Number of  reported results'], label='Historical Data')
plt.plot(data.index[train_size + lag:], predicted, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Reported Results')
plt.legend()
plt.savefig('plot/lstm_all.pdf')
plt.show()

# 使用最后的滞后窗口预测未来 90 天
future_steps = 90  # 预测未来的天数
last_window = X[-1:]  # 最后的滞后窗口
predictions = []

for i in range(future_steps):
    # 预测未来值
    y_future = model.predict(last_window)
    # 收集预测值
    predictions.append(y_future[0, 0])
    # 更新滞后窗口
    last_window = np.append(last_window, y_future, axis=1)
    last_window = last_window[:, 1:]

# 将预测结果反归一化
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(data.index, values, label='Historical Data', color='blue')
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
plt.plot(future_dates, predictions, label='Future Predictions', color='orange')
plt.title('Historical Data and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Number of Reported Results')
plt.legend()
plt.grid()
plt.savefig('plot/lstm_future.pdf')
plt.show()

# 可视化（局部）
plt.figure(figsize=(10, 6))
plt.plot(data.index[-120:], values[-120:], label='Historical Data', color='blue')
plt.plot(future_dates, predictions, label='Future Predictions', color='orange')
plt.title('Historical Data and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Number of Reported Results')
plt.legend()
plt.grid()
plt.savefig('plot/lstm_future_part.pdf')
plt.show()
