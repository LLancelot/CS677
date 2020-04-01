import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
# ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df_Friday = pd.read_csv(output_file)
# df_Friday = df[df['Weekday'] == 4]

# df1 = df_Friday[df_Friday['Year'] == 2018]
df2 = df_Friday[df_Friday['Year'] == 2019]
#
# label_lst = df1['label'].tolist()
# mean_lst = df1['mean_return'].tolist()
# volatility_lst = df1['volatility'].tolist()
# week_id = [x for x in range(1, len(label_lst)+1)]
#


plt.title("year2 three strategies")



data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df2)+1)],
        'Price': df2['Adj Close'].tolist(),
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Price','Label','X','Y']
)


label_lst = df2['label'].tolist()


week_id = [x for x in range(1, len(label_lst) + 1)]

train_x = np.asarray(week_id)
train_y = data2['Price'].values

week_range = [k for k in range(5,13)]
degree_range = [1,2,3]


def calculate(degree, week):
    # degree = [1,2,3]
    # week = [5,6,...,12]
    global assigned_label
    correct_num = 0
    pred_label = data2['Label'].values.copy()
    for index, row in data2.iterrows():
        if index + week == len(data2):
            # print("d=",degree," w=",week," accuracy="
            #       ,round(correct_num/(len(data2)-week),3), sep='')
            # break
            return pred_label


        weights = np.polyfit(train_x[index:index+week], train_y[index:index+week],degree)
        model = np.poly1d(weights)

        if model(index+week) > train_y[index+week-1]:
            assigned_label = "green"
            if assigned_label == label_lst[index+week]:
                correct_num += 1
        elif model(index+week) < train_y[index+week-1]:
            assigned_label = "red"
            if assigned_label == label_lst[index+week]:
                correct_num += 1
        elif model(index+week) == train_y[index+week-1]:
            assigned_label = label_lst[index+week-1]
            if assigned_label == label_lst[index+week]:
                correct_num += 1

        pred_label[index+week] = assigned_label


fri_price_lst = df2['Adj Close'].tolist()
days_fri = len(fri_price_lst)

def trading(predicted):

    label_lst = predicted


    myTrading_value = 0
    hold_strategy_value = 0
    myTrading_shares = 0
    hold_strategy_shares = 0

    cur_label = predicted[0]
    days_fri = len(fri_price_lst)
    begin = 0

    lst_daily_value = []
    lst_hold_value = []
    for i in range(days_fri - 1):
        if label_lst[i + 1] == 'green':
            myTrading_value += 100
            myTrading_shares += 100 / fri_price_lst[i]
            begin = i
            break

    for i in range(begin, days_fri - 1):
        if label_lst[i + 1] == 'green':
            if cur_label == 'green':
                lst_daily_value.append(myTrading_value)
                continue
            elif cur_label == 'red':
                # we should buy all i have
                myTrading_shares += myTrading_value / fri_price_lst[i]
                lst_daily_value.append(myTrading_value)
                cur_label = 'green'

        elif label_lst[i + 1] == 'red':
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
    return lst_daily_value


x_axis = [x for x in range(0, days_fri)]
p1 = trading(predicted=calculate(1,8))
p2 = trading(predicted=calculate(2,9))
p3 = trading(predicted=calculate(3,12))

plt.title("Comparison of three strategies with Degree and Week")
plt.plot(x_axis, p1)
plt.plot(x_axis, p2)
plt.plot(x_axis, p3)
plt.legend(['degree=1, W=8','degree=2, W=9', 'degree=3, W=12'])
plt.xlabel("Week")
plt.ylabel("Portfolio Value ($)")

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
# for week in range(len(training_data)):
#     x1 = training_data[week][0]
#     x2 = training_data[week][1]
#     w1 = weights[0][0]
#     w2 = weights[0][1]
#     func = log_reg_classifier.intercept_[0] + w1 * x1 + w2 * x2
#     y_pred = 1 / (1 + np.exp(-func))
#     print(week,predicted[week], y_pred)



