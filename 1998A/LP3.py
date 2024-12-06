import matplotlib.pyplot as plt
import numpy as np
from pulp import *

# 参数
M = 1
s = 0.05

r = [9.6, 18.5, 49.4, 23.9, 8.1, 14, 40.7, 31.2, 33.6, 36.8, 11.8, 9, 35, 9.4, 15]
r = [i / 100 for i in r]
# Adding a random value (positive or negative) to r
r = [i + np.random.normal(0, 0.1) for i in r]

q = [42, 54, 60, 42, 1.2, 39, 68, 33.4, 53.3, 40, 31, 5.5, 46, 5.3, 23]
q = [i / 100 for i in q]
# Adding a random value (positive or negative) to r
# q = [i + np.random.normal(0, 0.2) for i in r]
p = [2.1, 3.2, 6.0, 1.5, 7.6, 3.4, 5.6, 3.1, 2.7, 2.9, 5.1, 5.7, 2.7, 4.5, 7.6]
p = [i / 100 for i in p]
u = [181, 407, 428, 549, 270, 397, 178, 220, 475, 248, 195, 320, 267, 328, 131]

size = 15


def get_profit(lambda_risk):
    # 定义问题
    model = LpProblem("Portfolio_Optimization", LpMaximize)
    # 定义变量
    x = LpVariable.dicts("x", range(size), lowBound=0)
    # 手续费
    w = LpVariable.dicts("w", range(size), lowBound=0)
    # 风险
    z = LpVariable("z", lowBound=0)

    # 第二问：加入风险参数
    # 目标函数：最大化收益与最小化风险的权衡
    model += (1 - lambda_risk) * (lpSum([r[i] * x[i] - w[i] for i in range(size)]) + s * (
            M - lpSum(x[i] for i in range(size)))) - lambda_risk * z, "Objective"

    # 约束
    model += lpSum([x[i] for i in range(size)]) <= M, "Budget"
    for i in range(size):
        model += x[i] >= 0
        model += w[i] >= p[i] * x[i]
        # model += w[i] >= p[i] * u[i]
        model += z >= q[i] * x[i]

    # 求解
    model.solve()

    total_investment = 0
    for i in range(size):
        total_investment += x[i].value()
    bank = M - total_investment

    # 输出结果
    return model.objective.value(), bank


risks = np.linspace(0, 1, 100)
result = np.array([get_profit(risk) for risk in risks])

profits = result[:, 0]
bank = result[:, 1]

plt.plot(risks, profits)
plt.xlabel("Risk Weight")
plt.ylabel("Profit")
plt.title("Profit vs Risk Weight")
# plt.savefig("riskweight_profit.pdf")
plt.savefig("error_3.pdf")
plt.show()

plt.scatter(risks, bank)
plt.xlabel("Risk Weight")
plt.ylabel("Bank Balance")
plt.title("Bank Balance vs Risk Weight")
# plt.savefig("scatter.pdf")
plt.show()
