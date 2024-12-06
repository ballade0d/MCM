import matplotlib.pyplot as plt
import numpy as np
from pulp import *

# 参数
M = 1
s = 0.05

r = [0.28, 0.21, 0.23, 0.25]
q = [0.025, 0.015, 0.055, 0.026]
p = [0.01, 0.02, 0.045, 0.065]
u = [103, 198, 52, 40]

size = 4


def get_profit(max_risk):
    # 定义变量
    x = LpVariable.dicts("x", range(size), lowBound=0)
    # 手续费
    w = LpVariable.dicts("w", range(size), lowBound=0)
    # 风险
    z = LpVariable("z", lowBound=0)

    model = LpProblem("Portfolio_Optimization", LpMaximize)

    # 目标函数
    model += lpSum([r[i] * x[i] - w[i] for i in range(size)]) + s * (
            M - lpSum(x[i] for i in range(size))), "Objective"

    # 约束
    model += lpSum([x[i] for i in range(size)]) <= M, "Budget"
    for i in range(size):
        model += x[i] >= 0
        model += w[i] >= p[i] * x[i]
        # model += w[i] >= p[i] * u[i]
        model += z >= q[i] * x[i]

    model += z <= max_risk  # 第一问：限制风险的情况

    # 求解
    model.solve()

    # 输出结果
    return model.objective.value()


x = np.linspace(0, 0.03, 100)
y = [get_profit(i) for i in x]

plt.plot(x, y)
plt.title("Risk-Return")
plt.xlabel("Risk")
plt.ylabel("Profit")
plt.savefig("fix_risk.pdf")
plt.show()
