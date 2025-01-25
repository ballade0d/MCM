import pandas as pd

hosts = pd.read_csv('data/summerOly_hosts.csv')
athletes = pd.read_csv('data/summerOly_athletes.csv')

years = hosts['Year'][-15:]
print(years)

countries = athletes['NOC'].unique()
country_record = {}

for country in countries:
    record = []
    for year in years:
        medal = athletes[(athletes['NOC'] == country) & (athletes['Year'] == year)]['Medal'].unique()
        if 'Gold' in medal or 'Silver' in medal or 'Bronze' in medal:
            record.append('Yes')
        else:
            record.append('No')
    country_record[country] = record

for country, record in country_record.items():
    start = False
    for i in range(1, len(record)):
        if record[i] == 'Yes':
            start = True
        if start and record[i] == 'No':
            break
    if start:
        print(country)
        break


