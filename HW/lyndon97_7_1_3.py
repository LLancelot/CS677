# best w = 26 in year 2018.

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

df = pd.read_csv(ticker_file)
df2019 = df[df['Year'] == 2019]

dailyPrice_2019 = df2019['Adj Close'].tolist()

data2 = pd.DataFrame({
    'id_day':[x for x in range(1, len(df2019)+1)],
    'close_price': dailyPrice_2019},
    columns = ['id_day','close_price']
)

w = 26
profit = 0
buy_cost = 0
ss_cost = 0
buy_share = 0
ss_share = 0
pos = "no"
close_time = 0
long_time = 0
short_time = 0

for index, row in data2.iterrows():
    if index + w == len(data2):
        break


    model = LinearRegression()
    X = data2[index:index + w]['id_day'].values
    y = data2[index:index + w]['close_price'].values
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)
    model.fit(X, y)
    predict_price = model.predict([X[-1] + 1])
    if predict_price > y[-1]:  # LONG
        if pos == 'no':
            buy_cost += 100
            buy_share += 100 / y[-1]
            pos = 'long'
            long_time += 1
        elif pos == 'short':
            profit += ss_cost - ss_share * y[-1]
            ss_cost, ss_share = 0, 0
            pos = 'no'
            close_time += 1
        elif pos == 'long':
            continue

    elif predict_price < y[-1]:  # SHORT
        if pos == 'no':
            ss_cost += 100
            ss_share += 100 / y[-1]
            pos = 'short'
            short_time += 1
        elif pos == 'long':
            profit += buy_share * y[-1] - buy_cost
            buy_share, buy_cost = 0, 0
            pos = 'no'
            close_time += 1
        elif pos == 'short':
            continue


    # last day
print("Long transaction times in year2:",long_time)
print("Short transaction times in year2:",short_time)

