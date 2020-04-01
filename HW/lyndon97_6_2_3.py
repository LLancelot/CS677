import os
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

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
close_price = df1['Adj Close'].tolist()
week_id = [x for x in range(1, len(label_lst) + 1)]

# data1 = pd.DataFrame(
#     {'week_id': [x for x in range(1, len(df1) + 1)],
#      'Price': close_price,
#      'Label': label_lst,
#      'X': mean_lst,
#      'Y': volatility_lst},
#     columns=['week_id', 'Price','Label', 'X', 'Y']
# )


data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df2)+1)],
        'Price': df2['Adj Close'].tolist(),
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Price','Label','X','Y']
)



# print(data1)

train_x = np.asarray(week_id)
train_y = data2['Price'].values

week_range = [k for k in range(5,13)]
degree_range = [1,2,3]

label_lst = df2['label'].tolist()

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

'''
d=1, w=8
d=2, w=9
d=3, w=12
'''
best_w = [8,9,12]
# print(sum(calculate(1,8) == data2['Label'].values))
# print(sum(calculate(2,9) == data2['Label'].values))
# print(sum(calculate(3,12) == data2['Label'].values))

matrix = confusion_matrix(y_true=data2['Label'].values, y_pred=calculate(1,8))
print("----Confusion Matrix----, degree=1, week=8")
print(matrix)
print()

matrix = confusion_matrix(y_true=data2['Label'].values, y_pred=calculate(2,9))
print("----Confusion Matrix----, degree=2, week=9")
print(matrix)
print()

matrix = confusion_matrix(y_true=data2['Label'].values, y_pred=calculate(3,12))
print("----Confusion Matrix----, degree=3, week=12")
print(matrix)
