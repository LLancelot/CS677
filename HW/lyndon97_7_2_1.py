'''
compute the contributions of µ and σ for logistic regression,
Euclidean kNN and (degree 1) linear model. Summarize
them in a table and discuss your findings
'''
import os
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from texttable import Texttable
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
week_id = [x for x in range(1, len(label_lst)+1)]

data1 = pd.DataFrame(
    {'id':[x for x in range(1, len(df1)+1)],
     'Label': label_lst,
     'Price': close_price,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id','Price','Label','X','Y']
)

data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df1)+1)],
        'Label': df2['label'].tolist(),
        'Price': df1['Adj Close'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Price','Label','X','Y']
)

# we take k=5 in kNN
k=5
# Accuracy(mu, sigma)
X_values = data1[['X','Y']].values
y_values = data1[['Label']].values
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_values,np.ravel(y_values))

new_instance = data2[['X','Y']].values
prediction = knn_classifier.predict(new_instance)
correct_num = sum(prediction == np.asarray(df2['label'].tolist()))
acc_X = accuracy_score(y_true=data2[['Label']].values, y_pred=prediction)

# Accuracy(sigma), remove mu
X_values = data1[['Y']].values
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_values,np.ravel(y_values))
new_instance = data2[['Y']].values
prediction = knn_classifier.predict(new_instance)
correct_num = sum(prediction == np.asarray(df2['label'].tolist()))
acc_sigma = accuracy_score(y_true=data2[['Label']].values, y_pred=prediction)

# Accuracy(mu), remove sigma
X_values = data1[['X']].values
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_values,np.ravel(y_values))
new_instance = data2[['X']].values
prediction = knn_classifier.predict(new_instance)
correct_num = sum(prediction == np.asarray(df2['label'].tolist()))
acc_mu = accuracy_score(y_true=data2[['Label']].values, y_pred=prediction)

# print("KNN:",acc_X,acc_mu,acc_sigma)

# Accuracy(mu,sigma) for logistic regression
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

acc_logreg_X = accuracy_score(year2_Label.flatten(),predicted)
# print(acc_logreg_X)

# Accuracy(sigma), remove mu
year1_ = data1[['Y']].values
year1_Label = data1['Label'].values

year2_ = data2[['Y']].values
year2_Label = data2['Label'].values

scaler = StandardScaler()
scaler.fit(year1_)
year1_ = scaler.transform(year1_)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(year1_,np.ravel(year1_Label))

training_data = year2_
predicted = log_reg_classifier.predict(training_data)
# correct_num = sum(predicted == np.asarray(label_lst))
weights = log_reg_classifier.coef_
w1_str = str(round(weights[0][0],2))
# w2_str = str(round(weights[0][1],2))
func_intercept_str = str(round(log_reg_classifier.intercept_[0],2))

acc_logreg_sigma = accuracy_score(year2_Label.flatten(),predicted)
# print(acc_logreg_sigma)

# Accuracy(mu), remove sigma
year1_ = data1[['X']].values
year1_Label = data1['Label'].values

year2_ = data2[['X']].values
year2_Label = data2['Label'].values

scaler = StandardScaler()
scaler.fit(year1_)
year1_ = scaler.transform(year1_)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(year1_,np.ravel(year1_Label))

training_data = year2_
predicted = log_reg_classifier.predict(training_data)
# correct_num = sum(predicted == np.asarray(label_lst))
weights = log_reg_classifier.coef_
w1_str = str(round(weights[0][0],2))
# w2_str = str(round(weights[0][1],2))
func_intercept_str = str(round(log_reg_classifier.intercept_[0],2))

acc_logreg_mu = accuracy_score(year2_Label.flatten(),predicted)
# print(acc_logreg_mu)

# linear regression, degree=1, (mu, sigma) -> price
w_range = [x for x in range(5,13)]


def get_best_week(degree, week,train_x,train_y):
    # degree = [1,2,3]
    # week = [5,6,...,12]
    correct_num = 0
    for index, row in df1.iterrows():
        if index + week == len(df1):
            # print("d=",degree," w=",week," accuracy="
            #       ,round(correct_num/(len(df1)-week),3), sep='')
            return round(100*(correct_num/(len(df1)-week)),2)


        reg = LinearRegression().fit(train_x[index:index+week], train_y[index:index+week])
        predict_linear_price = reg.predict([train_x[index+week]])


        if predict_linear_price > train_y[index+week-1]:
            assigned_label = "green"
            if assigned_label == label_lst[index+week]:
                correct_num += 1
        elif predict_linear_price < train_y[index+week-1]:
            assigned_label = "red"
            if assigned_label == label_lst[index+week]:
                correct_num += 1
        elif predict_linear_price == train_y[index+week-1]:
            assigned_label = label_lst[index+week-1]
            if assigned_label == label_lst[index+week]:
                correct_num += 1

lin_performance_by_week = []
for week in w_range:
    lin_performance_by_week.append(get_best_week(1,week,data1[['X','Y']].values,data1[['Price']].values))

# bestWeek = 6 for linear reg
bestWeek = lin_performance_by_week.index(max(lin_performance_by_week)) + 5

def test_linear_model(week, train_x, train_y, test_x, test_y):
    correct_num = 0
    label_ = df2['label'].tolist()
    for index, row in df1.iterrows():
        if index + week == len(df1):
            # print("d=",degree," w=",week," accuracy="
            #       ,round(correct_num/(len(df1)-week),3), sep='')
            return round(100 * (correct_num / (len(df1) - week)), 2)

        reg = LinearRegression().fit(train_x[index:index + week], train_y[index:index + week])
        predict_linear_price = reg.predict([test_x[index + week]])

        if predict_linear_price > test_y[index + week - 1]:
            assigned_label = "green"
            if assigned_label == label_[index + week]:
                correct_num += 1
        elif predict_linear_price < test_y[index + week - 1]:
            assigned_label = "red"
            if assigned_label == label_[index + week]:
                correct_num += 1
        elif predict_linear_price == test_y[index + week - 1]:
            assigned_label = label_[index + week - 1]
            if assigned_label == label_[index + week]:
                correct_num += 1

acc_lin_both=test_linear_model(6, train_x=data1[['X','Y']].values,train_y=data1[['Price']].values,test_x=data2[['X','Y']].values, test_y=data2[['Price']].values)
acc_lin_sigma=test_linear_model(6, train_x=data1[['Y']].values,train_y=data1[['Price']].values,test_x=data2[['Y']].values,test_y=data2[['Price']].values)
acc_lin_mu=test_linear_model(6,train_x=data1[['X']].values,train_y=data1[['Price']].values,test_x=data2[['X']].values,test_y=data2[['Price']].values)

# summary the result.
cont_knn_mu = str(round((acc_X - acc_sigma)*100,2))+"%"
cont_knn_sigma = str(round((acc_X - acc_mu)*100,2))+"%"
cont_log_mu = str(round((acc_logreg_X - acc_logreg_sigma)*100,2))+"%"
cont_log_sigma = str(round((acc_logreg_X - acc_logreg_mu)*100,2))+"%"
cont_lin_mu = str(round((acc_lin_both - acc_lin_sigma),2))+"%"
cont_lin_sigma = str(round((acc_lin_both - acc_lin_mu),2))+"%"

res_table = pd.DataFrame(
    {
        'Method':['KNN','Logistic Regression','Linear Regression'],
        'Contribution of μ':[cont_knn_mu,cont_log_mu,cont_lin_mu],
        'Contribution of σ':[cont_knn_sigma,cont_log_sigma,cont_lin_sigma]
    }
)
tb=Texttable()
tb.set_cols_align(['l','r','r'])
tb.set_cols_dtype(['t','i','i'])
tb.header(res_table.columns)
tb.add_rows(res_table.values,header=False)

print(tb.draw())



