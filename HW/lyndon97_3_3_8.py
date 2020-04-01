'''7. what are the top 5 most popular items for each day of the week?
does this list stays the same from day to day?'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')
df = pd.read_csv(file)
# out_file = os.path.join(input_dir, 'res_1.csv')
# grouped = df.groupby(['Weekday','Item']).count()
# print(grouped['Year'])

item_count = df.groupby(['Weekday','Item'],sort=True).agg(count = ('Item','count'))
# item_count.to_csv(os.path.join(input_dir, 'res_2.csv'))
res = item_count.sort_values(['Weekday','count'], ascending=True).groupby('Weekday').head(5)
print("The top 5 least popular items for each day of the week is listed below:")

print(res)
print('The least popular items are quiet different from day to day.')