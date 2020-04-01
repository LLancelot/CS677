'''
compute 'cent' digit's distribution
'''
import os
import pandas as pd

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_five_year.csv')

df = pd.read_csv(ticker_file)
price_list = df['Adj Close'].tolist()

number_vector = [0] * 10

# convert all the closing prices into string
for i in range(len(price_list)):
    string = "{0:.2f}".format(price_list[i])
    price_list[i] = string

for str_price in price_list:
    last_digit = int(str_price[-1])
    number_vector[last_digit] += 1

error_vector = []
for num in number_vector:
    error_vector.append(num - 100)


print("The most frequent digit is", number_vector.index(max(number_vector)))
# print("The least frequent digit is", number_vector.index(min(number_vector)))


