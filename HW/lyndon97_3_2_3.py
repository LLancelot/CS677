'''
compute RMSE for model_1 and model_2
'''
import pandas as pd
import os
import math
from collections import Counter


wd = os.getcwd()
input_dir = wd
csv_file = os.path.join(input_dir, 'online_retail.csv')
lead_digit_lst = []
distribution_lst = [0]*10
m1 = [0.111] * 9
m_Bernford = [0.301,0.176,0.125,0.097,0.079,0.067,0.058,0.051,0.046]

df = pd.read_csv(csv_file)
unitPrice_lst = df['UnitPrice'].tolist()
for i in range(len(unitPrice_lst)):
    unitPrice_lst[i] = str(unitPrice_lst[i])
    lead_digit_lst.append(int(unitPrice_lst[i][0]))
sum_ = 0
lead_counter = Counter(lead_digit_lst)
for key in lead_counter.keys():
    distribution_lst[key] = lead_counter[key]


size = sum(distribution_lst) - distribution_lst[0]

distribution_lst = distribution_lst[1:]

# convert d_lst into fraction
for i in range(len(distribution_lst)):
    distribution_lst[i] = distribution_lst[i] / size

error_abs_m1 = []
error_abs_m2 = []

for i in range(len(distribution_lst)):
    error_abs_m1.append(abs(distribution_lst[i] - m1[i]))
    error_abs_m2.append(abs(distribution_lst[i] - m_Bernford[i]))

print("The RMSE for model1 is",round(math.sqrt(sum(error_abs_m1) / len(distribution_lst)),2))
print("The RMSE for model2 is",round(math.sqrt(sum(error_abs_m2) / len(distribution_lst)),2))

