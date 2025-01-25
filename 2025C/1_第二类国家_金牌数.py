import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional, LSTM, Dense, Multiply

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

country = 'United States'

# 数据读取和处理
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')
hosts = pd.read_csv('data/summerOly_hosts.csv')

medal_counts = medal_counts[medal_counts['NOC'] == country]
years = medal_counts['Year'].values

X_extra = np.array([])
for row in hosts.iterrows():
    year = row[1]['Year']
    if year not in years:
        continue
    host = row[1]['Host']
    if host == country:
        X_extra = np.append(X_extra, 1)
    else:
        X_extra = np.append(X_extra, 0)
X_extra = X_extra.reshape(-1, 1)
X = medal_counts['Gold'].values.reshape(-1, 1)
X = np.hstack((X, X_extra))


# 创建滞后序列函数
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(X)

lag = 6
X, y = create_lagged_data(scaled_values, lag)

# 数据划分
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
host_input = Input(shape=(1, 1))
attention = Dense(1, activation='sigmoid')(host_input)
weighted_input = Multiply()([input_layer, attention])
lstm = Bidirectional(LSTM(64, return_sequences=False))(weighted_input)
lstm = Dense(32)(lstm)
output_layer = Dense(1)(lstm)

model = Model(inputs=[input_layer, host_input], outputs=[output_layer])
model.compile(optimizer='adam', loss='mae')

# 提取主场变量
host_train = X_train[:, -1, 1]  # 提取每个序列最后一个时间步的主场变量
host_test = X_test[:, -1, 1]  # 同上

# 确保主场变量形状正确
host_train = host_train.reshape(-1, 1)  # (样本数, 1)
host_test = host_test.reshape(-1, 1)  # (样本数, 1)

# 训练模型
history = model.fit(
    [X_train, host_train], [y_train],
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_data=([X_test, host_test], y_test),
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(factor=0.8, patience=10)],
)

# 预测
predicted = model.predict([X_test, host_test])  # 双输入
predicted = predicted.reshape(-1, 1)
predicted = scaler.inverse_transform(np.hstack((predicted, np.zeros_like(predicted))))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1)))))[:, 0]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual Values', color='blue')
plt.plot(predicted, label='Predicted Values', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Medals')
plt.legend()
plt.savefig('plot/预测金牌数_局部_' + country + '.pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(medal_counts['Year'], medal_counts['Gold'], label='Historical')
plt.plot(medal_counts['Year'][-len(predicted):], predicted, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Medals')
for i in range(len(years)):
    host = X_extra[i][0]
    if host == 1:
        plt.axvline(x=years[i], color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('plot/预测金牌数_全局_' + country + '.pdf')
plt.show()

# 预测2028，host=True
future_year = 2028
future_host = 1  # 美国是东道主
last_sequence = scaled_values[-lag:]  # 获取最后一个滞后序列

# 创建2028年的输入数据
future_input = np.hstack((last_sequence[:, 0].reshape(-1, 1), last_sequence[:, 1].reshape(-1, 1)))  # 滞后序列
future_input[-1, 1] = future_host  # 设置最后一个输入的 host 信息
future_input = future_input.reshape(1, lag, X.shape[2])  # 调整成模型需要的输入形状

# 主场输入
future_host_input = np.array([[future_host]])  # 主场变量，形状 (1, 1)

# 修改 predict 方法
future_prediction = model.predict([future_input, future_host_input])  # 双输入
future_prediction_scaled = future_prediction.reshape(-1, 1)

# 反归一化预测值
future_prediction_actual = scaler.inverse_transform(
    np.hstack((future_prediction_scaled, np.zeros_like(future_prediction_scaled))))[:, 0]

print(f"Predicted medal count for the US in {future_year} (as host): {future_prediction_actual[0]:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(medal_counts['Year'], medal_counts['Gold'], label='Historical')
plt.plot(medal_counts['Year'][-len(predicted):], predicted, label='Predicted', linestyle='--')
plt.scatter(future_year, future_prediction_actual, color='red', label='2028 Prediction')
plt.title('Predicted Values')
plt.xlabel('Time')
plt.ylabel('Number of Medals')
plt.legend()
plt.savefig('plot/预测2028金牌数_' + country + '.pdf')
plt.show()
