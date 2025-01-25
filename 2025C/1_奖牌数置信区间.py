import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional, LSTM, Dense, Multiply
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # 用于显示进度条

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
X = medal_counts['Total'].values.reshape(-1, 1)
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
X_all, y_all = create_lagged_data(scaled_values, lag)

# 数据划分
train_size = int(len(X_all) * 0.8)
X_train_all, X_test = X_all[:train_size], X_all[train_size:]
y_train_all, y_test = y_all[:train_size], y_all[train_size:]

# 提取主场变量
host_test = X_test[:, -1, 1]  # 测试集的主场变量
host_test = host_test.reshape(-1, 1)  # (样本数, 1)

# 定义bootstrap的次数
n_iterations = 50
bootstrap_predictions = []

# 设置未来预测的参数
future_year = 2028
future_host = 1  # 美国是东道主

# 迭代进行 bootstrap
for i in tqdm(range(n_iterations), desc='Bootstrap Iterations'):
    # 重采样（有放回）
    indices = np.random.choice(len(X_train_all), size=len(X_train_all), replace=True)
    X_train = X_train_all[indices]
    y_train = y_train_all[indices]

    # 构建模型
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
    host_train = host_train.reshape(-1, 1)  # (样本数, 1)

    # 训练模型
    history = model.fit(
        [X_train, host_train], y_train,
        epochs=50,  # 可以适当减少 epoch 数以加快计算
        batch_size=16,
        verbose=0,  # 设为 0 以减少输出
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # 创建2028年的输入数据
    last_sequence = scaled_values[-lag:]  # 获取最后一个滞后序列
    future_input = np.hstack((last_sequence[:, 0].reshape(-1, 1), last_sequence[:, 1].reshape(-1, 1)))
    future_input[-1, 1] = future_host  # 设置最后一个输入的 host 信息
    future_input = future_input.reshape(1, lag, X_all.shape[2])  # 调整成模型需要的输入形状

    # 主场输入
    future_host_input = np.array([[future_host]])  # 主场变量，形状 (1, 1)

    # 预测
    future_prediction = model.predict([future_input, future_host_input], verbose=0)
    future_prediction_scaled = future_prediction.reshape(-1, 1)

    # 反归一化预测值
    future_prediction_actual = scaler.inverse_transform(
        np.hstack((future_prediction_scaled, np.zeros_like(future_prediction_scaled))))[:, 0]

    # 保存预测结果
    bootstrap_predictions.append(future_prediction_actual[0])

# 计算置信区间
alpha = 0.05  # 95% 置信区间
lower_bound = np.percentile(bootstrap_predictions, 100 * (alpha / 2))
upper_bound = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2))
mean_prediction = np.mean(bootstrap_predictions)

print(f"Predicted medal count for the US in {future_year} (as host): {mean_prediction:.2f}")
print(f"95% confidence interval: [{lower_bound:.2f}, {upper_bound:.2f}]")

# 绘制预测分布
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_predictions, bins=30, edgecolor='k', alpha=0.7)
plt.title('Bootstrap Predictions Distribution for 2028')
plt.xlabel('Predicted Medal Count')
plt.ylabel('Frequency')
plt.axvline(x=lower_bound, color='red', linestyle='--', label=f'Lower 95% CI ({lower_bound:.2f})')
plt.axvline(x=upper_bound, color='green', linestyle='--', label=f'Upper 95% CI ({upper_bound:.2f})')
plt.axvline(x=mean_prediction, color='blue', linestyle='-', label=f'Mean Prediction ({mean_prediction:.2f})')
plt.legend()
plt.savefig('plot/预测2028奖牌数分布_' + country + '.pdf')
plt.show()
