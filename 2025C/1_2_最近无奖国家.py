import pandas as pd

# Define the years of interest
years = [2024, 2020, 2016, 2012, 2008, 2004, 2000, 1996, 1992, 1988]

# Load the dataset
athletes = pd.read_csv('data/summerOly_athletes.csv')

# Filter the dataset for the specified years
filtered_data = athletes[athletes['Year'].isin(years)]

# Get countries that have won medals
countries_with_medals = filtered_data[filtered_data['Medal'] != "No medal"]['NOC'].unique()

# Get all countries in the dataset during these years
all_countries = filtered_data['NOC'].unique()

# Identify countries with no medals
countries_without_medals = set(all_countries) - set(countries_with_medals)

# Display the result
print("Countries with no medals in the specified years:")
print(countries_without_medals)
