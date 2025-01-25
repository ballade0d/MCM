import pandas as pd

athletes = pd.read_csv("data/summerOly_athletes.csv")

# 去除未获奖的运动员
athletes = athletes[athletes["Medal"] != "No medal"]
# 获取每个国家每个Sport的奖牌数
medals = athletes.groupby(["NOC", "Sport"]).size().unstack().fillna(0)
medals.to_csv("data/medals_per_sport.csv")
# 除以每个国家的总奖牌数
medals = medals.div(medals.sum(axis=1), axis=0)
# 保存结果
medals.to_csv("data/medals_percentage_per_sport.csv")
