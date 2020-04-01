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

data1 = pd.DataFrame(
    {'week_id': [x for x in range(1, len(df1) + 1)],
     'Price': close_price,
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns=['week_id', 'Price','Label', 'X', 'Y']
)

year1_re_and_vo = data1[['X', 'Y']].values
year1_Label = data1['Label'].values

# print(data1)

train_x = np.asarray(week_id)
train_y = np.asarray(close_price)

week_range = [k for k in range(5,13)]
degree_range = [1,2,3]

for degree in degree_range:
    for week in week_range:
        pass

def calculate(degree, week):
    # degree = [1,2,3]
    # week = [5,6,...,12]
    correct_num = 0
    for index, row in df1.iterrows():
        if index + week == len(df1):
            # print("d=",degree," w=",week," accuracy="
            #       ,round(correct_num/(len(df1)-week),3), sep='')
            return round(100*(correct_num/(len(df1)-week)),2)


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

result1 = []
result2 = []
result3 = []
for i in degree_range:
    for j in week_range:
        if i == 1:
            result1.append(calculate(i,j))
        elif i == 2:
            result2.append(calculate(i,j))
        elif i == 3:
            result3.append(calculate(i,j))


plt.title("Degrees and Weeks for Prediction Accuracies")
plt.xlabel("period of weeks")
plt.ylabel("accuracy (%)")
plt.plot(week_range, result1, c='red')
plt.plot(week_range, result2, c='blue')
plt.plot(week_range, result3, c='yellow')

plt.legend(['Degree=1', 'Degree=2','Degree=3'])
plt.show()

'''
d=1, w=8
d=2, w=9
d=3, w=12
'''