'''
what's the most profitable time
- hour
- day of a week
- period
'''

import os
import pandas as pd
from collections import Counter



wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(file)

hour_profit_lst = [0]*24
weekday_profit_dict = {'Monday':0, 'Tuesday':0, 'Wednesday':0, 'Thursday':0, 'Friday':0,
                      'Saturday':0, 'Sunday':0}
period_profit_dict = {'afternoon': 0, 'morning': 0, 'evening': 0, 'night': 0}
for index, row in df.iterrows():
    hour_profit_lst[row['Hour']] += row['Item_Price']
    weekday_profit_dict[row['Weekday']] += row['Item_Price']
    period_profit_dict[row['Period']] += row['Item_Price']


print("The most profitable hour is", hour_profit_lst.index(max(hour_profit_lst)),", with profit",round(max(hour_profit_lst),2))
print("The most profitable day of the week is", max(weekday_profit_dict.items(), key=
                                                    lambda k:k[1])[0],", with profit",round(max(weekday_profit_dict.items(), key=
                                                    lambda k:k[1])[1],2))
print("The most profitable period is", max(period_profit_dict.items(), key =
                                           lambda k:k[1])[0],", with profit",round(max(period_profit_dict.items(), key =
                                           lambda k:k[1])[1],2))
