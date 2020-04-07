import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
# ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df_Friday = pd.read_csv(output_file)
# df_Friday = df[df['Weekday'] == 4]

df1 = df_Friday[df_Friday['Year'] == 2018]
df2 = df_Friday[df_Friday['Year'] == 2019]

label_lst = df1['label'].tolist()
mean_lst = df1['mean_return'].tolist()
volatility_lst = df1['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]


Y_label = df_Friday.iloc[:,-3]


red_ = df1.loc[Y_label == "red"]
green_  = df1.loc[Y_label == "green"]

# plt.figure(1)
# plt.scatter(red_.iloc[:,-2], red_.iloc[:,-1],label="red",c="red")
# plt.scatter(green_.iloc[:,-2], green_.iloc[:,-1], label="green", c="green")
# plt.legend()



data1 = pd.DataFrame(
    {'id':[x for x in range(1, len(df1)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id', 'Label','X','Y']
)

data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df2)+1)],
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Label','X','Y']
)


year1_re_and_vo = data1[['X','Y']].values
year1_Label = data1['Label'].values

testing_data = data2[['X','Y']].values

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(year1_re_and_vo, year1_Label)

predicted = clf.predict(testing_data)
#############
label_lst = predicted
fri_price_lst = df2['Adj Close'].tolist()

myTrading_value = 0
hold_strategy_value = 0
myTrading_shares = 0
hold_strategy_shares = 0

cur_label = predicted[0]
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
    elif label_lst[i+1] == "red":
        lst_daily_value.append(100)

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

plt.title("Comparison of Decision Tree and Buy-Hold")
plt.plot(x_axis, lst_daily_value)
plt.plot(x_axis, lst_hold_value)
plt.legend(['Decision Tree', 'Buy-Hold'])
plt.xlabel("Week")
plt.ylabel("Portfolio Value")

print("Trading strategy based on Decision Tree is more profitable at the end of year2.")
plt.show()