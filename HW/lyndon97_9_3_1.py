import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
            plt.plot(x_axis, error_rate, c=(random.random(),random.random(),random.random()))
            legend.append("d = "+str(d))
            error_rate = []

print("The best combination of N and d:","N =",min_n,", d =",min_d)
# print("The best combination of N and d: N=4, d=2")

plt.title("Error Rate of N and d")
plt.xlabel("The range of N")
plt.ylabel("Error Rate (%)")
plt.legend(legend)
plt.show()

# print(best_predicted)
'''
since it's random, the best combination that I got is N=5, d=1.
After setseed():

predicted labels:
['red' 'green' 'red' 'red' 'green' 'red' 'green' 'green' 'red' 'green'
 'green' 'red' 'green' 'green' 'red' 'red' 'green' 'red' 'green' 'red'
 'red' 'green' 'green' 'red' 'red' 'green' 'green' 'red' 'red' 'red'
 'green' 'red' 'red' 'green' 'red' 'green' 'green' 'red' 'red' 'green'
 'red' 'red' 'green' 'green' 'green' 'red' 'green' 'red' 'green' 'green'
 'green']
 
 '''





