'''
implement Polynomial SVM (degree = 2) classifier
'''

import os
import pandas as pd
import numpy as np
from sklearn import svm

wd = os.getcwd()
ticker = 'LIN'
input_dir = wd
# ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)

df1 = df[df['Year'] == 2018]
df2 = df[df['Year'] == 2019]

label_lst = df1['label'].tolist()
mean_lst = df1['mean_return'].tolist()
volatility_lst = df1['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

data1 = pd.DataFrame(
    {'week_id':[x for x in range(1, len(df1)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['week_id', 'Label','X','Y']
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
testing_Label = data2['Label'].values

svm_classifier_poly = svm.SVC(kernel='poly', degree=2)
svm_classifier_poly.fit(year1_re_and_vo, year1_Label)
predicted_poly = svm_classifier_poly.predict(testing_data)
accuracy_poly = round(sum(predicted_poly==data2['Label'].values.ravel())/len(data2['Label'].values.ravel())*100,2)

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(year1_re_and_vo, year1_Label)
predicted = svm_classifier.predict(testing_data)
accuracy_linear = round(sum(predicted==data2['Label'].values.ravel())/len(data2['Label'].values.ravel())*100,2)

print("Accuracy of Polynomial SVM (degree=2) classifier for year 2:", str(accuracy_poly)+"%")


if accuracy_poly > accuracy_linear:
    print("The accuracy using Polynomial SVM (degree=2) is better than Linear SVM.")
else:
    print("The accuracy using Polynomial SVM (degree=2) is worse than Linear SVM.")





















