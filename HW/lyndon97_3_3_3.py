import os
import pandas as pd
from collections import Counter



wd = os.getcwd()
input_dir = wd
file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

df = pd.read_csv(file)

item_lst = df['Item'].tolist()
item_couter = Counter(item_lst)


print("The most popular item is", max(item_couter.items(), key=
                                      lambda k:k[1])[0])
min_value = min(item_couter.values())
str_min_item = ""
for key in item_couter.keys():
    if item_couter[key] == min_value:
        str_min_item += key + ', '
str_min_item = str_min_item[:-2]

print("The least popular items are",str_min_item)

