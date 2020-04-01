import os
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

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

df_Friday = df_Friday[df_Friday['Year'] == 2018]
# df_Friday = df_Friday[df_Friday['Year'] == 2019]

label_lst = df_Friday['label'].tolist()
mean_lst = df_Friday['mean_return'].tolist()
volatility_lst = df_Friday['volatility'].tolist()
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


red_ = df_Friday.loc[Y_label == "red"]
green_  = df_Friday.loc[Y_label == "green"]

plt.figure(1)
plt.scatter(red_.iloc[:,-2], red_.iloc[:,-1],label="red",c="red")
plt.scatter(green_.iloc[:,-2], green_.iloc[:,-1], label="green", c="green")
plt.legend()

# build our model

data1 = pd.DataFrame(
    {'id':[x for x in range(1, len(df_Friday)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id', 'Label','X','Y']
)

year1_re_and_vo = data1[['X','Y']].values
year1_Label = data1['Label'].values

scaler = StandardScaler()
scaler.fit(year1_re_and_vo)
year1_re_and_vo = scaler.transform(year1_re_and_vo)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(year1_re_and_vo,np.ravel(year1_Label))

training_data = year1_re_and_vo
predicted = log_reg_classifier.predict(training_data)
correct_num = sum(predicted == np.asarray(label_lst))
weights = log_reg_classifier.coef_
w1_str = str(round(weights[0][0],2))
w2_str = str(round(weights[0][1],2))
func_intercept_str = str(round(log_reg_classifier.intercept_[0],2))
print("Equation for logistic regression:")
print("h(x) = ",func_intercept_str+" "+w1_str+"*x1"+" "+w2_str+"*x2")

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



