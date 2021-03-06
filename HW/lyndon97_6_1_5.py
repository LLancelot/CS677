import os
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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



plt.title("year1")
# for i in range(len(label_lst)):
#     if label_lst[i] == 'green':
#         plt.scatter(mean_lst[i],volatility_lst[i],c = 'green')
#     elif label_lst[i] == 'red':
#         plt.scatter(mean_lst[i],volatility_lst[i],c = 'red')
#
# for xi, yi, wi, label in zip(mean_lst,volatility_lst,week_id,label_lst):
#     if label == 'green':
#         plt.scatter(xi,yi,c='green')
#         # plt.annotate(str(wi), xy=(xi,yi))
#     elif label == 'red':
#         plt.scatter(xi, yi, c='red')
#         # plt.annotate(str(wi), xy=(xi, yi))


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

year2_re_and_vo = data2[['X','Y']].values
year2_Label = data2['Label'].values

scaler = StandardScaler()
scaler.fit(year1_re_and_vo)
year1_re_and_vo = scaler.transform(year1_re_and_vo)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(year1_re_and_vo,np.ravel(year1_Label))

training_data = year2_re_and_vo
predicted = log_reg_classifier.predict(training_data)
# correct_num = sum(predicted == np.asarray(label_lst))
weights = log_reg_classifier.coef_
w1_str = str(round(weights[0][0],2))
w2_str = str(round(weights[0][1],2))
func_intercept_str = str(round(log_reg_classifier.intercept_[0],2))


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

print("According to the plot, the label-based strategy is more profitable.")
plt.show()

#
# print(training_data)
# print(predicted)
# print(correct_num, log_reg_classifier.score(year1_re_and_vo,year1_Label))
# print(weights)
# print(accuracy_score(year1_Label.flatten(),predicted))
# print()

# x1=training_data[2][0]
# x2=training_data[2][1]
# w1 = weights[0][0]
# w2 = weights[0][1]
# func = log_reg_classifier.intercept_[0] + w1*x1 + w2*x2
# y_pred=1/(1+np.exp(-func))
# print(predicted[0],y_pred)
# plt.show()
for week in range(len(training_data)):
    x1 = training_data[week][0]
    x2 = training_data[week][1]
    w1 = weights[0][0]
    w2 = weights[0][1]
    func = log_reg_classifier.intercept_[0] + w1 * x1 + w2 * x2
    y_pred = 1 / (1 + np.exp(-func))
    print(week,predicted[week], y_pred)



