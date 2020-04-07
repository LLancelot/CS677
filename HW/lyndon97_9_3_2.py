'''
since it's random, the best combination that I got is N=5, d=1 after setseed().

predicted labels:
['red' 'green' 'red' 'red' 'green' 'red' 'green' 'green' 'red' 'green'
 'green' 'red' 'green' 'green' 'red' 'red' 'green' 'red' 'green' 'red'
 'red' 'green' 'green' 'red' 'red' 'green' 'green' 'red' 'red' 'red'
 'green' 'red' 'red' 'green' 'red' 'green' 'green' 'red' 'red' 'green'
 'red' 'red' 'green' 'green' 'green' 'red' 'green' 'red' 'green' 'green'
 'green']

 '''
import os
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

class_labels_dict = {'green':1, 'red': 0}
label_color_dict = {1:'green', 0:'red'}
data1['class_labels'] = data1['Label'].map(class_labels_dict)

def calculate_random_forest(n, d):
    np.random.seed(1)
    predicted = []
    model = RandomForestClassifier(n_estimators=n, max_depth=d, criterion='entropy')
    model.fit(year1_re_and_vo, year1_Label)

    return [np.mean(model.predict(testing_data) != testing_Label), model.predict(testing_data)]

error_rate = []
min_error_rate = 1.0
min_d, min_n = 0, 0
best_predicted = []
x_axis = [x for x in range(1,11)]
legend = []
for d in range(1,6):
    for n in range(1,11):
        val = calculate_random_forest(n,d)[0]
        error_rate.append(100*val)
        if min_error_rate > val:
            min_error_rate = val

            min_d, min_n = d, n
            best_predicted = calculate_random_forest(n,d)[1]
        if n == 10:
            # plt.plot(x_axis, error_rate, c=(random.random(),random.random(),random.random()))
            legend.append("d = "+str(d))
            error_rate = []



predicted = best_predicted
matrix = confusion_matrix(data2['Label'].values.ravel(), predicted)
print("Using optimal values from year1, the confusion matrix for year2:")
print(matrix)

