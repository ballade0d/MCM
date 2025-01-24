import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

medal_counts = pd.read_csv('../data/summerOly_medal_counts.csv')
hosts = pd.read_csv('../data/summerOly_hosts.csv')

# calculate weighted historical medal counts
# USA only
medal_counts = medal_counts[medal_counts['NOC'] == 'United States']
# check host
medal_counts['host'] = 0
# reset index
medal_counts = medal_counts.reset_index(drop=True)
for i in range(len(medal_counts)):
    year = medal_counts['Year'][i]
    host = hosts[hosts['Year'] == year]['Host'].values[0]
    country = medal_counts['NOC'][i]
    if host == country:
        medal_counts.loc[i, 'host'] = 1

# use bp model to predict total medal counts
X = medal_counts[['Year', 'host']]
y = medal_counts['Total']

from sklearn.preprocessing import StandardScaler

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

# Input Layer and Hidden Layers
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 64 neurons, ReLU activation
model.add(Dense(32, activation='relu'))  # 32 neurons, ReLU activation

# Output Layer
model.add(Dense(1, activation='linear'))  # 1 neuron, linear activation for regression

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the Model on the Test Set
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print("Mean Absolute Error on Test Set:", mae)

# Predict 2028
future_data = pd.DataFrame({'Year': [2028], 'host': [1]})  # Host country in 2028
future_data_scaled = scaler.transform(future_data)  # Apply the same scaling
future_prediction = model.predict(future_data_scaled)

# visualize
plt.plot(medal_counts['Year'], medal_counts['Total'], label='Historical')
plt.plot([2028], future_prediction, 'ro', label='Prediction')
plt.show()
