import numpy as np
import pandas as pd

data = pd.read_excel("2021MCMProblemC_DataSet.xlsx")

pos = np.column_stack((data['Longitude'], data['Latitude']))

x_min, x_max = data['Longitude'].min(), data['Longitude'].max()
y_min, y_max = data['Latitude'].min(), data['Latitude'].max()

x_min_border = x_min - abs(x_min - x_max) * 0.1
x_max_border = x_max + abs(x_min - x_max) * 0.1
y_min_border = y_min - abs(y_min - y_max) * 0.1
y_max_border = y_max + abs(y_min - y_max) * 0.1

# 获取格子范围
print(x_min_border, x_max_border)
print(y_min_border, y_max_border)
