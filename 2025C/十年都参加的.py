import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

years = [2024, 2020, 2016, 2012, 2008, 2004, 2000, 1996, 1992, 1988]
medal_counts = pd.read_csv('data/summerOly_medal_counts.csv')

# 获取所有年份都参加奥运会的国家
countries = []
for year in years:
    countries.append(set(medal_counts[medal_counts['Year'] == year]['NOC']))
countries = list(set.intersection(*countries))

print(countries)
