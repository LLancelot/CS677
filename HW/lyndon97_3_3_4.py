import os
import pandas as pd
from collections import Counter



wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(file)

weekday_bas_dict = {'Monday':0, 'Tuesday':0, 'Wednesday':0, 'Thursday':0, 'Friday':0,
                      'Saturday':0, 'Sunday':0}
# begin = 0
# lst = []
# for index, row in df.iterrows():
#     if index<len(df)-1:
#         if df['Weekday'][index] != df['Weekday'][index+1]:
#             lst = df[begin:index+1]['Item'].tolist()
#             cnt = lst.count("Coffee")
#             begin = index
#             if cnt > weekday_bas_dict[df['Weekday'][index]]:
#                 weekday_bas_dict[df['Weekday'][index]] = cnt

df = df[df['Item'] == 'Coffee']
print(df.groupby(['Weekday','Year','Month','Day'])['Transaction'].count().groupby('Weekday').max() / 50)

print("Monday needs:","2 barristas")
print("Tuesday needs:","1 barristas")
print("Wednesday needs:","1 barristas")
print("Thursday needs:","1 barristas")
print("Friday needs:","2 barristas")
print("Saturday needs:","2 barristas")
print("Sunday needs:","2 barristas")

