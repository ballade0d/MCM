from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

X = [2024, 2020, 2016, 2012, 2008, 1996, 1992, 1988, 1984, 1976, 1972, 1968, 1964, 1960, 1956, 1952, 1948, 1936, 1932,
     1928, 1924, 1920, 1912, 1908]
y = [0.5775862068965517, 0.4217687074829932, 0.4512820512820513, 0.41630901287553645, 0.5606060606060606,
     0.2914572864321608, 0.41798941798941797, 0.37037037037037035, 0.5869565217391305, 0.33636363636363636,
     0.4215686274509804, 0.3404255319148936, 0.4473684210526316, 0.3888888888888889, 0.36363636363636365, 0.5,
     0.35714285714285715, 0.19767441860465115, 0.5405405405405406, 0.26, 0.5636363636363636, 0.5263157894736842,
     0.47619047619047616, 0.64]

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
plt.title('Old Athlete Winning Rate Prediction')
plt.legend()
plt.show()
