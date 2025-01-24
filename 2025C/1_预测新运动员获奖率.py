from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

X = [2024, 2020, 2016, 2012, 2008, 1996, 1992, 1988, 1984, 1976, 1972, 1968, 1964, 1960, 1956, 1952, 1948, 1936, 1932,
     1928, 1924, 1920, 1912, 1908]
y = [0.32041343669250644, 0.4074074074074074, 0.3416666666666667, 0.35135135135135137, 0.390745501285347,
     0.37723214285714285, 0.301994301994302, 0.31043956043956045, 0.5539112050739958, 0.3487544483985765,
     0.3016949152542373, 0.3218390804597701, 0.40298507462686567, 0.34673366834170855, 0.37264150943396224,
     0.41706161137440756, 0.4482758620689655, 0.2556390977443609, 0.3264781491002571, 0.2775330396475771,
     0.5165289256198347, 0.45864661654135336, 0.423841059602649, 0.3829787234042553]

# random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit([[x] for x in X_train], y_train)

# Evaluate the Model on the Test Set
score = model.score([[x] for x in X_test], y_test)
print("R^2 Score on Test Set:", score)
# MSE
y_pred = model.predict([[x] for x in X_test])
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Predict 2028
future_prediction = model.predict([[2028]])
print("2028预测：", future_prediction)
# visualize
plt.plot(X, y, label='Historical')
plt.plot([2028], future_prediction, 'ro', label='Prediction')
plt.title('New Athlete Winning Rate Prediction')
plt.legend()
plt.show()
