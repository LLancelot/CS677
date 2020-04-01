'''6. does this coﬀee shop make more money from selling drinks or from selling food?'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')
df1 = pd.read_csv(file)
df2 = pd.read_excel('item.xlsx')

merge_output = pd.merge(df1, df2, on=['Item'], how='left').fillna(0)
drinks_profit = 0
food_profit = 0
for index, row in merge_output.iterrows():
    if row['sorted']=='drinks':
        drinks_profit += row['Item_Price']
    elif row['sorted']=='food':
        food_profit += row['Item_Price']
print("The money from selling drinks is %.2f"% round(drinks_profit,2))
print("The money from selling food is %.2f"% round(food_profit,2))
print("So this coffee shop makes more money from selling drinks." if drinks_profit >= food_profit else "So this coffee shop makes more money from selling food.")
