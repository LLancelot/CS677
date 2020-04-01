
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
label_lst = df_Friday['label'].tolist()
fri_price_lst = df_Friday['Adj Close'].tolist()

myTrading_value = 0
hold_strategy_value = 0
myTrading_shares = 0
hold_strategy_shares = 0

cur_label = 'green'
days_fri = len(fri_price_lst)
begin = 0

lst_daily_value = []
lst_hold_value = []
for i in range(days_fri-1):
    if label_lst[i+1] == 'green':
        myTrading_value += 100
        myTrading_shares += 100 / fri_price_lst[i]
        begin = i
        break

for i in range(begin, days_fri-1):
    if label_lst[i+1] == 'green':
        if cur_label == 'green':
            lst_daily_value.append(myTrading_value)
            continue
        elif cur_label == 'red':
            # we should buy all i have
            myTrading_shares += myTrading_value / fri_price_lst[i]
            lst_daily_value.append(myTrading_value)
            cur_label = 'green'

    elif label_lst[i+1] == 'red':
        if cur_label == 'red':
            lst_daily_value.append(myTrading_value)
            continue

        elif cur_label == 'green':
            # sell shares
            myTrading_value = myTrading_shares * fri_price_lst[i]
            myTrading_shares = 0
            lst_daily_value.append(myTrading_value)
            cur_label = 'red'
lst_daily_value.append(myTrading_value)

# buy-hold strategy
hold_strategy_shares = 100 / fri_price_lst[begin]

for i in range(days_fri):
    lst_hold_value.append(hold_strategy_shares*fri_price_lst[i])



x_axis = [x for x in range(0, days_fri)]
plt.title("Comparison of two strategies")
plt.plot(x_axis, lst_daily_value)
plt.plot(x_axis, lst_hold_value)
plt.legend(['Label-based', 'Buy-Hold'])
plt.xlabel("Week")
plt.ylabel("Portfolio Value")
plt.show()
