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
res = item_count.sort_values(['Weekday','count'], ascending=False).groupby('Weekday').head(5)
print("The top 5 most popular items for each day of the week is listed below:")
print(res)

print("Based on the result, Tuesday and Thursday are the same, plus Sunday and Wednesday are the same.")
print('Coffee, bread, tea are top three every day of the week.')

# s=df['Weekday'].groupby(df['Item']).value_counts()
#
# dfs = pd.read_csv(out_file)
# wd = ["Sunday","Saturday","Friday","Thursday","Tuesday","Wednesday","Monday"]
# data_mon = dfs[dfs['Weekday'] == 'Monday']
# data_tue = dfs[dfs['Weekday'] == 'Tuesday']
# data_wed = dfs[dfs['Weekday'] == 'Wednesday']
# data_thr = dfs[dfs['Weekday'] == 'Thursday']
# data_fri = dfs[dfs['Weekday'] == 'Friday']
# data_sat = dfs[dfs['Weekday'] == 'Saturday']
# data_sun = dfs[dfs['Weekday'] == 'Sunday']
#
# # print(len(data_mon)+len(data_tue)+len(data_wed)+len(data_thr)+len(data_fri)+len(data_sat)+len(data_sun))
#
# print("Mon",data_mon.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Tue',data_tue.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Wed',data_wed.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Thu',data_thr.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Fri',data_fri.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Sat',data_sat.sort_values('count', ascending=False)[:5]["Item"].tolist())
# print('Sun',data_sun.sort_values('count', ascending=False)[:5]["Item"].tolist())
#
#
# print("Mon",data_mon.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Tue',data_tue.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Wed',data_wed.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Thu',data_thr.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Fri',data_fri.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Sat',data_sat.sort_values('count', ascending=False)[-5:]["Item"].tolist())
# print('Sun',data_sun.sort_values('count', ascending=False)[-5:]["Item"].tolist())
#
