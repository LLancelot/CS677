import os
import pandas as pd
import numpy as np

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df_smoke = df[df['smoker'] == 'Yes']
sum_smoke = sum(df_smoke['size'])

percent_smoke = 100 * sum_smoke / sum(df['size'])
print("There is",str(round(percent_smoke,2))+"%","of people who are smoking.")