from pulp import *

# 定义问题
model = LpProblem("Portfolio_Optimization", LpMaximize)

# 参数
M = 1
s = 0.05

r = [0.28, 0.21, 0.23, 0.25]
q = [0.025, 0.015, 0.055, 0.026]
p = [0.01, 0.02, 0.045, 0.065]
u = [103, 198, 52, 40]

size = 4

# 定义变量
x = LpVariable.dicts("x", range(size), lowBound=0)
# 手续费
w = LpVariable.dicts("w", range(size), lowBound=0)
# 风险
z = LpVariable("z", lowBound=0)

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

model += z <= 0.006  # 第一问：限制风险的情况

# 求解
model.solve()

# 输出结果
total_investment = 0
for i in range(size):
    print(f"Investment in asset {i}: {x[i].value()}")
    total_investment += x[i].value()
    print(f"Transaction cost in asset {i}: {w[i].value()}")
    print()
bank = M - total_investment
print(f"Total investment: {total_investment}")
print(f"Bank balance: {bank}")
print(f"Total profit: {model.objective.value()}")
print(f"Total risk (max risk chosen): {z.value()}")
