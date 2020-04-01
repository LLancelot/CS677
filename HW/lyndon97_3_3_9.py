'''9. how many drinks are there per transaction?'''

import os
import pandas as pd

wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')
df1 = pd.read_csv(file)
df2 = pd.read_excel('item.xlsx')

merge_output = pd.merge(df1, df2, on=['Item'], how='left').fillna(0)

trans_lst = merge_output['Transaction'].unique().tolist()

trans_drinks_lst = {}.fromkeys(trans_lst,0)
for index, row in merge_output.iterrows():
    if row['sorted']=='drinks':
        trans_drinks_lst[row['Transaction']] += 1
for key in trans_drinks_lst.keys():
    print("Transaction %d " % key,"have %d drinks." % trans_drinks_lst[key])