import pandas as pd

data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
frequency = pd.read_excel('词频2.xlsx')

# 将frequency最后的百分号去掉
frequency['frequency'] = frequency['frequency'].str[:-1].astype(float)

try_time = data[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']]

from sklearn.model_selection import train_test_split

# 多输出回归
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# X为将word和frequency合并的数据，y为try_time
X = pd.merge(data, frequency, on='Word')[['Word', 'frequency']]
y = try_time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
multi_target_forest_reg = MultiOutputRegressor(forest_reg, n_jobs=-1)
multi_target_forest_reg.fit(X_train['frequency'].values.reshape(-1, 1), y_train)
y_pred = multi_target_forest_reg.predict(X_test['frequency'].values.reshape(-1, 1))

# 评估
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(mse)

# 画出测试集中部分数据的预测值和真实值

import matplotlib.pyplot as plt

# 从测试集中随机选取一个样本
import random

sample_index = random.randint(0, len(X_test) - 1)
X_test_sample = X_test.iloc[sample_index]
y_test_sample = y_test.iloc[sample_index]
y_pred_sample = y_pred[sample_index]

plt.plot(range(1, 8), y_test_sample, 'r-', label='True')
plt.plot(range(1, 8), y_pred_sample, 'b-', label='Predict')
plt.title('True and Predict of one sample')
plt.legend()
plt.savefig('plot/sample.pdf')
plt.show()

# 预测新的单词
my_word = 'eerie'
my_frequency = 0.0002396

pred = multi_target_forest_reg.predict([[my_frequency]])
print(pred)

plt.plot(range(1, 8), pred[0], 'b-', label='Predict')
plt.title('Predict of word EERIE')
plt.legend()
plt.savefig('plot/eerie.pdf')
plt.show()

print(pred)