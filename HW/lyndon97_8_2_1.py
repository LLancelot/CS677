'''
Implement a Naive Bayesian classifier. For each week, your
feature set is (µ; σ) for that week. Use your labels (you will
have 52 labels per year for each week) from year 1 to train your
classifier and predict labels for year 2. Use Gaussian Naive
Bayesian (this is the default).

implement a Gaussian naive bayesian classifier and compute
its accuracy for year 2
'''

import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

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
NB_classifier = GaussianNB().fit(year1_re_and_vo, year1_Label.ravel())
testing_data = data2[['X','Y']].values
predicted = NB_classifier.predict(testing_data)
accuracy = round(sum(predicted==data2['Label'].values.ravel())/len(data2['Label'].values.ravel())*100,2)
print('The accuracy for year 2 is','%s%%'%accuracy)


