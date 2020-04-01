import os
import pandas as pd
import numpy as np

wd = os.getcwd()
input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

df_smoke = df[df['smoker'] == 'Yes']
df_nonsmoke = df[df['smoker'] == 'No']

# calculate cor [total_bill, tip] among smoke/non-smoke
cor1 = df_smoke['tip'].corr(df_smoke['total_bill'])
cor2 = df_nonsmoke['tip'].corr(df_nonsmoke['total_bill'])

smoke_tips_mean = df_smoke['tip_percent'].mean()
nonsmoke_tips_mean = df_nonsmoke['tip_percent'].mean()


r2_cor1 = cor1**2
r2_cor2 = cor2**2

print("We use r-square to evaluate the what percentage of the total "
      "\nvariation in the variables is explained by this correlation.")
print("The r-square of smokers is 67.5% and nonsmokers is 23.8%,"
      "\nwhich means the relationship between tips and meal prices are "
      "\nstrongly associated among non-smokers than those smokers."
      "\nThus the correlation between smokers and non-smokers are different.")

# print(smoke_tips_mean,nonsmoke_tips_mean)
print("Also, the average tips percentage of smokers is 16.32%, and non-smokers is 15.93%,"
      "\nwhich means smokers will give more tips than non-smokers.")
