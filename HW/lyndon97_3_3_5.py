'''5.  divide all items in 3 groups (drinks, food, unknown).
 What is the average price of a drink and a food item?
'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')
df1 = pd.read_csv(file)
df2 = pd.read_excel('item.xlsx')

merge_output = pd.merge(df1, df2, on=['Item'], how='right').fillna(0)
Item_price_sorted = merge_output[['Item','Item_Price','sorted']].drop_duplicates(subset = None, keep = 'first')
average_price = Item_price_sorted.groupby(['sorted']).agg({'Item_Price': 'mean'})

print(average_price)