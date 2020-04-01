'''
Model_1: abs(real - m1) / m1
         abs(m2 - m1) / m1
         0;

Model_2: abs(real - m2) / m2
         abs(m1 - m2) / m2
         0;

'''


import pandas as pd
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

# calculate relative errors
# for model_1

r_error_real_m1 = []
r_error_m1_m1 = [0] * 9
r_error_m2_m1 = []

r_error_real_m2 = []
r_error_m2_m2 = [0] * 9
r_error_m1_m2 = []

for i in range(len(distribution_lst)):
    r_error_real_m1.append(abs(distribution_lst[i] - m1[i]) / m1[i])
    r_error_m2_m1.append(abs(m_Bernford[i] - m1[i]) / m1[i])
    r_error_real_m2.append(abs(distribution_lst[i] - m_Bernford[i]) / m_Bernford[i])
    r_error_m1_m2.append(abs(m_Bernford[i] - m1[i]) / m_Bernford[i])

print(r_error_real_m1)
print(r_error_m2_m1)
print(r_error_real_m2)
print(r_error_m1_m2)

barWidth = 0.25
r1 = np.arange(len(distribution_lst))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


# plt.title("Histogram of Three Distributions")

plt.bar(r1, r_error_real_m1, color='#FF0000', width=barWidth, edgecolor='white', label='real distribution')
plt.bar(r2, r_error_m1_m1, color='#00FF00', width=barWidth, edgecolor='white', label='model_1')
plt.bar(r3, r_error_m2_m1, color='#0000FF', width=barWidth, edgecolor='white', label='model_2')

plt.title("Relative Error for Model 1")
plt.xlabel('digit', fontweight='bold')
plt.ylabel("Relative Error", fontweight = 'bold')
plt.xticks([r + barWidth for r in range(len(distribution_lst))], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])


plt.legend()
plt.show()


plt.bar(r1, r_error_real_m2, color='#FF0000', width=barWidth, edgecolor='white', label='real distribution')
plt.bar(r2, r_error_m2_m2, color='#00FF00', width=barWidth, edgecolor='white', label='model_1')
plt.bar(r3, r_error_m1_m2, color='#0000FF', width=barWidth, edgecolor='white', label='model_2')

plt.title("Relative Error for Model 2")
plt.xlabel('digit', fontweight='bold')
plt.ylabel("Relative Error", fontweight = 'bold')
plt.xticks([r + barWidth for r in range(len(distribution_lst))], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])


plt.legend()
plt.show()