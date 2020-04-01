import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

df = pd.read_csv(ticker_file)
df2018 = df[df['Year'] == 2018]

dailyPrice_2018 = df2018['Adj Close'].tolist()

data1 = pd.DataFrame({
    'id_day':[x for x in range(1, len(df2018)+1)],
    'close_price': dailyPrice_2018},
    columns = ['id_day','close_price']
)


profit = 0
long_profit = 0
short_profit = 0
buy_cost = 0
ss_cost = 0
buy_share = 0
ss_share = 0
pos = "no"
close_time = 0
long_time = 0
short_time = 0

w=26

for index, row in data1.iterrows():
    if index + w == len(data1):
        break

    model = LinearRegression()
    X = data1[index:index+w]['id_day'].values
    y = data1[index:index+w]['close_price'].values
    X = X.reshape(len(X),1)
    y = y.reshape(len(y),1)
    model.fit(X,y)
    predict_price = model.predict([X[-1]+1])
    if predict_price > y[-1]: # LONG
        if pos == 'no':
            buy_cost += 100
            buy_share += 100 / y[-1]
            pos = 'long'
            long_time += 1
        elif pos == 'short':
            profit += ss_cost - ss_share * y[-1]
            short_profit += ss_cost - ss_share * y[-1]
            ss_cost, ss_share = 0, 0
            pos = 'no'
            close_time +=1
        elif pos == 'long':
            continue

    elif predict_price < y[-1]: # SHORT
        if pos == 'no':
            ss_cost += 100
            ss_share += 100 / y[-1]
            pos = 'short'
            short_time += 1
        elif pos == 'long':
            profit += buy_share * y[-1] - buy_cost
            long_profit += buy_share * y[-1] - buy_cost
            buy_share, buy_cost = 0, 0
            pos = 'no'
            close_time += 1
        elif pos == 'short':
            continue

res1 = round(long_profit[0]/long_time,2)
res2 = round(short_profit[0] /short_time,2)
print("The result with w=26 in year1:")
print("The average P/L per long position is $",res1," in year1",sep='')
print("The average P/L per short position is $",res2," in year1",sep='')
print("The average number of days for long position and short position is",int((long_time+short_time)/2))
print()
'''
details:
The average P/L per long position is $0.89 in year1
The average P/L per short position is $0.44 in year1
The average number of days for long position and short position is 18.0
---------------------
The average P/L per long position is $1.5 in year2
The average P/L per short position is $-0.27 in year2
The average number of days for long position and short position is 16.5

'''
print("My conclusion:")
print("Generally, the average P/L per long position in year2 is higher than year1,"
      "\nbut the average P/L per short position in year2 is less than year1,"
      "\nwhich is negative. In terms of the number of days for long position"
      "\nand short position, both year1 and year2 have similar result.")
