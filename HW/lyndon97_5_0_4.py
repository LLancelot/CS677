'''
analyse your trading with these labels in 2019
'''

import os
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')

df = pd.read_csv(ticker_file)

df_Friday = df[df['Weekday'] == 4]

df_Friday = df_Friday[df_Friday['Year'] == 2019]

# copy the label from predicted.
label_lst = ['green', 'green', 'green', 'red', 'green', 'red', 'green', 'green', 'red', 'green', 'green', 'red', 'green', 'green', 'red', 'red', 'green', 'green', 'green', 'green', 'red', 'green', 'green', 'green', 'red', 'green', 'green', 'red', 'red', 'red', 'green', 'red', 'red', 'green', 'green', 'green', 'green', 'red', 'red', 'green', 'green', 'green', 'green', 'green', 'green', 'red', 'green', 'green', 'green', 'green', 'green']
fri_price_lst = df_Friday['Adj Close'].tolist()

myTrading_value = 0
myTrading_shares = 0

weekly_shares = []

cur_label = 'green'
days_fri = len(fri_price_lst)
begin = 0

lst_daily_value = []

for i in range(days_fri-1):
    if label_lst[i+1] == 'green':
        myTrading_value += 100
        myTrading_shares += 100 / fri_price_lst[i]
        begin = i
        weekly_shares.append(myTrading_shares)
        break

for i in range(begin, days_fri-1):
    if label_lst[i+1] == 'green':
        if cur_label == 'green':
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            continue
        elif cur_label == 'red':
            # we should buy all i have
            myTrading_shares += myTrading_value / fri_price_lst[i]
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            cur_label = 'green'

    elif label_lst[i+1] == 'red':
        if cur_label == 'red':
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            continue

        elif cur_label == 'green':
            # sell shares
            myTrading_value = myTrading_shares * fri_price_lst[i]
            myTrading_shares = 0
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            cur_label = 'red'
lst_daily_value.append(myTrading_value)

print("Analyze the trading result based on the predicted labels in 2019,"
      "\nthe final account balance is $",round(myTrading_value,2),", start with $100.",sep='')
x_axis = [x for x in range(0, days_fri)]
plt.title("Growth of My Account")
plt.plot(x_axis, lst_daily_value)

plt.legend(['Label-based'])
plt.xlabel("Week")
plt.ylabel("Account Balance")
plt.show()