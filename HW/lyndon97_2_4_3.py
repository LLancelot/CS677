'''
compute 'cent' digit's distribution
'''
import os
import pandas as pd
import numpy as np
from tabulate import tabulate

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_five_years.csv')

df = pd.read_csv(ticker_file)
price_list = df['Adj Close'].tolist()

number_vector = [0] * 10
number_vector_15 = [0] * 10
number_vector_16 = [0] * 10
number_vector_17 = [0] * 10
number_vector_18 = [0] * 10
number_vector_19 = [0] * 10

# convert all the closing prices into string
for i in range(len(price_list)):
    string = "{0:.2f}".format(price_list[i])
    price_list[i] = string

for str_price in price_list:
    last_digit = int(str_price[-1])
    number_vector[last_digit] += 1

sum_all = sum(number_vector)
for i in range(len(number_vector)):
    number_vector[i] = number_vector[i] / sum_all


error_vector = []
for num in number_vector:
    error_vector.append(abs(num - 0.1))



# print(number_vector)
# # print(sorted(error_vector))
# print(error_vector)
# print("The most frequent digit is", number_vector.index(max(number_vector)))
# print("The least frequent digit is", number_vector.index(min(number_vector)))
print("From 2015 ~ 2019")
print("(a) Four years max absolute error:", max(error_vector))
print("(b) Four years median absolute error:", np.median(error_vector))
print("(c) Four years mean absolute error:", np.mean(error_vector))
print("(d) Four years root mean squared error:", np.sqrt(np.mean(error_vector)))

df2015 = df[df['Year'] == 2015]
df2016 = df[df['Year'] == 2016]
df2017 = df[df['Year'] == 2017]
df2018 = df[df['Year'] == 2018]
df2019 = df[df['Year'] == 2019]

price_list_15 = df2015['Adj Close'].tolist()
price_list_16 = df2016['Adj Close'].tolist()
price_list_17 = df2017['Adj Close'].tolist()
price_list_18 = df2018['Adj Close'].tolist()
price_list_19 = df2019['Adj Close'].tolist()

# formating price_list
for i in range(len(price_list_15)):
    string = "{0:.2f}".format(price_list_15[i])
    price_list_15[i] = string

for i in range(len(price_list_16)):
    string = "{0:.2f}".format(price_list_16[i])
    price_list_16[i] = string

for i in range(len(price_list_17)):
    string = "{0:.2f}".format(price_list_17[i])
    price_list_17[i] = string

for i in range(len(price_list_18)):
    string = "{0:.2f}".format(price_list_18[i])
    price_list_18[i] = string

for i in range(len(price_list_19)):
    string = "{0:.2f}".format(price_list_19[i])
    price_list_19[i] = string

# get count and distribution
for str_price in price_list_15:
    last_digit = int(str_price[-1])
    number_vector_15[last_digit] += 1

for str_price in price_list_16:
    last_digit = int(str_price[-1])
    number_vector_16[last_digit] += 1

for str_price in price_list_17:
    last_digit = int(str_price[-1])
    number_vector_17[last_digit] += 1

for str_price in price_list_18:
    last_digit = int(str_price[-1])
    number_vector_18[last_digit] += 1

for str_price in price_list_19:
    last_digit = int(str_price[-1])
    number_vector_19[last_digit] += 1


error_vector = []
error_vector_15 = []
error_vector_16 = []
error_vector_17 = []
error_vector_18 = []
error_vector_19 = []

sum_15 = sum(number_vector_15)
sum_16 = sum(number_vector_16)
sum_17 = sum(number_vector_17)
sum_18 = sum(number_vector_18)
sum_19 = sum(number_vector_19)


for i in range(len(number_vector_15)):
    number_vector_15[i] = number_vector_15[i] / sum_15

for i in range(len(number_vector_16)):
    number_vector_16[i] = number_vector_16[i] / sum_16

for i in range(len(number_vector_17)):
    number_vector_17[i] = number_vector_17[i] / sum_17

for i in range(len(number_vector_18)):
    number_vector_18[i] = number_vector_18[i] / sum_18

for i in range(len(number_vector_19)):
    number_vector_19[i] = number_vector_19[i] / sum_19


for num in number_vector_15:
    error_vector_15.append(abs(num - 0.1))
for num in number_vector_16:
    error_vector_16.append(abs(num - 0.1))
for num in number_vector_17:
    error_vector_17.append(abs(num - 0.1))
for num in number_vector_18:
    error_vector_18.append(abs(num - 0.1))
for num in number_vector_19:
    error_vector_19.append(abs(num - 0.1))


# print("From 2015 ~ 2019")
# print("(a) max absolute error:", max(error_vector))
# print("(b) median absolute error:", np.median(error_vector))
# print("(c) mean absolute error:", np.mean(error_vector))
# print("(d) root mean squared error:", np.sqrt(np.mean(error_vector)))


error_ = [error_vector_15,error_vector_16,error_vector_17,error_vector_18,error_vector_19]
res_max_error_vector = []
res_median_error_vector = []
res_mean_error_vector = []
res_RMSE_error_vector = []

for item in error_:
    res_max_error_vector.append(max(item))
    res_median_error_vector.append(np.median(item))
    res_mean_error_vector.append((np.mean(item)))
    res_RMSE_error_vector.append(np.sqrt(np.mean(item)))

data = pd.DataFrame({'Year':[2015,2016,2017,2018,2019],
                     'Max Absolute Error':res_max_error_vector,
                     'Median Absolute Error':res_median_error_vector,
                     'Mean Absolute Error':res_mean_error_vector,
                     'RMSE':res_RMSE_error_vector})
print(tabulate(data, headers='keys', tablefmt='psql'))
