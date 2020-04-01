import os
import pandas as pd
import numpy as np

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# calculate the correlation between tips and size of the group
cor_tip_size = df['tip_percent'].corr(df['size'])
# print(cor_tip_size)
print("Since the correlation coefficient between tips and size of the group"
      "\nis nearly -0.143, which is negative and the absolute value is close to zero."
      "\nIn other words, the relationship between tips percentage and size is too week.")