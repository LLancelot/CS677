import os
import pandas as pd
import numpy as np

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

print(df.groupby(['day','time'],as_index = False)['tip_percent'].agg({'tip_percent':'mean'}))

print("The highest tips in average day and time belongs to the lunch time on Friday")
# print("The highest tips among weekday is Saturday, for time is dinner.")
