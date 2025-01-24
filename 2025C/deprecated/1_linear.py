import pandas as pd

medal_counts = pd.read_csv('../data/summerOly_medal_counts.csv')
hosts = pd.read_csv('../data/summerOly_hosts.csv')

# calculate weighted historical medal counts
# USA only
medal_counts = medal_counts[medal_counts['NOC'] == 'United States']
# check host and weight
medal_counts['host'] = 0
medal_counts['weight'] = 0.0
# reset index
medal_counts = medal_counts.reset_index(drop=True)
for i in range(len(medal_counts)):
    year = medal_counts['Year'][i]
    host = hosts[hosts['Year'] == year]['Host'].values[0]
    country = medal_counts['NOC'][i]
    if host == country:
        medal_counts.loc[i, 'host'] = 1
    # loop all previous years
    weight = 0
    for j in range(i):
        weight += 1 / (i - j) * medal_counts['Total'][j]
    medal_counts.loc[i, 'weight'] = weight

# use lineary regression model to predict total medal counts
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = medal_counts[['Year', 'host', 'weight']]
y = medal_counts['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model on the Test Set
score = model.score(X_test, y_test)
print("R^2 Score on Test Set:", score)
# MSE
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)

# Predict 2028
weight = 0
for i in range(len(medal_counts)):
    weight += 1 / (len(medal_counts) - i) * medal_counts['Total'][i]
future_data = pd.DataFrame({'Year': [2028], 'host': [1], 'weight': [weight]})  # Host country in 2028
future_data_scaled = scaler.transform(future_data)  # Apply the same scaling
future_prediction = model.predict(future_data_scaled)
print("2028预测：", future_prediction)

# visualize
import matplotlib.pyplot as plt

plt.plot(medal_counts['Year'], medal_counts['Total'], label='Historical')
plt.plot([2028], future_prediction, 'ro', label='Prediction')
plt.legend()
plt.show()

# plot the test set
y_pred = model.predict(X_test)
# transform back
X_test = scaler.inverse_transform(X_test)
plt.scatter(X_test[:, 0], y_test, color='black')
plt.scatter(X_test[:, 0], y_pred, color='blue')
plt.show()
