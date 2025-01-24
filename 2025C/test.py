import pandas as pd
from matplotlib import pyplot as plt

count = pd.read_csv('data/summerOly_medal_counts.csv')

# get countries in 2024
countries = count[count['Year'] == 2024]

for country in countries['NOC']:
     # get count of United States
     t = count[count['NOC'] == country]
     # plot
     plt.plot(t['Year'], t['Total'])
     plt.xlabel('Year')
     plt.ylabel('Total Medal Count')
     plt.title('Total Medal Count of ' + country)
     plt.show()
