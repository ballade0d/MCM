import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
frequency = pd.read_excel('词频2.xlsx')

# 合并两个表
data = data.merge(frequency, on='Word')

# 将frequency最后的百分号去掉
data['frequency'] = data['frequency'].str[:-1].astype(float)
data['score'] = (0.5 * (1 * data['1 try'] + 2 * data['2 tries'] + 3 * data['3 tries'] + 4 * data['4 tries'] + 5 * data[
    '5 tries'] + 6 * data['6 tries']) + 0.5 * data['7 or more tries (X)']) / 100

# 对frequency log10
data['frequency'] = data['frequency'].apply(lambda x: np.log10(x))

# 画frequency与score的散点图
plt.scatter(data['frequency'], data['score'], s=5)
plt.xlabel('Frequency')
plt.ylabel('score')
plt.show()

# 画pearson correlation热力图
t = data[['score', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)', 'frequency']]

corr = t.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation')
plt.show()
