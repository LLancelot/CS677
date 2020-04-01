'''
what is the busiest (in terms of number of transactions)
- hour
- day of a week
- period

'''
# transactions from a bakery
import os
import pandas as pd
from collections import Counter



wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(file)

hour_lst = df['Hour'].tolist()

hour_counter = Counter(hour_lst)

max_hour = hour_counter[1]
k = None

for key in hour_counter.keys():
    if hour_counter[key] > max_hour:
        k = key
        max_hour = hour_counter[key]

print("The busiest hour is", k,"'o clock")

weekday_lst = df['Weekday'].tolist()
weekday_counter = Counter(weekday_lst)

max_weekday = weekday_counter['Sunday']
day = None
for key in weekday_counter.keys():
    if weekday_counter[key] > max_weekday:
        max_weekday = weekday_counter[key]
        day = key

print("The busiest day of the week is", day)

period_lst = df['Period'].tolist()
period_counter = Counter(period_lst)

max_period = period_counter['morning']
period = None

for key in period_counter.keys():
    if period_counter[key] > max_period:
        max_period = period_counter[key]
        period = key

print("The busiest period is", period)


