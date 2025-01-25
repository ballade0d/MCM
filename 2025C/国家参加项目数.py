import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
athletes = pd.read_csv('data/summerOly_athletes.csv')

# Filter data for the year 2024
athletes = athletes[athletes['Year'] == 2024]

# Initialize an empty DataFrame to store the results
data = pd.DataFrame(columns=['NOC', 'Sport', 'Medal'])

# Loop through each country and calculate the number of sports and medals
for country in athletes['NOC'].unique():
    sports = athletes[athletes['NOC'] == country]['Sport'].nunique()
    # Exclude 'No medal' entries
    medals = athletes[(athletes['NOC'] == country) & (athletes['Medal'] != 'No medal')]['Medal'].count()
    data = pd.concat([data, pd.DataFrame({'NOC': [country], 'Sport': [sports], 'Medal': [medals]})])

# drop countries with no medals
data = data[data['Medal'] > 0]
data = data.reset_index(drop=True)
data = data[['Sport', 'Medal']]

# draw correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between the number of sports and medals')
plt.show()
