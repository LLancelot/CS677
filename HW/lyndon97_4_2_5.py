
import os
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
friday_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')

df = pd.read_csv(ticker_file)
df1 = pd.read_csv(friday_file)
df_Friday = df[df['Weekday'] == 4]
label_lst = df_Friday['label'].tolist()
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

weekly_change = [0]
for i in range(1, len(lst_daily_value)):
    weekly_change.append(lst_daily_value[i] - lst_daily_value[i-1])
print("Since the value of my account was always increasing, the maximum duration of growing is 102 weeks.")
print("And thus there is no duration in weeks that my account was decreasing.")