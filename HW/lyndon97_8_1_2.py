import os
import pandas as pd
import numpy as np

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

print("Average tip percentage for each day of the week (%):")
print(df.groupby('day')['tip_percent'].mean())

print("Summary:")
print("The average tip percentage for Friday is 16.99%")
print("The average tip percentage for Saturday is 15.31%")
print("The average tip percentage for Sunday is 16.69%")
print("The average tip percentage for Thursday is 16.13%")
