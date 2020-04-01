import os
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']
df['meal_price'] = df['total_bill'] - df['tip']

cor = df["meal_price"].corr(df['tip_percent'])
# cor1 = pearsonr(df['meal_price'],df['tip_percent'])
# cor1 = cor1[0]
# print(cor1)
print("The correlation coefficient between meal price and tips is",round(cor,3))
