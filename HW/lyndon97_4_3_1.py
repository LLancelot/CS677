'''
Implement k-NN classifier For each week, your feature set is
(µ; σ) for that week. Use your labels (you will have 52 labels
per year for each week) from 2017 to train your classifier and
predict labels for 2018
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)

df_Friday = df[df['Weekday'] == 4]

plt.xlabel("The value of K")
plt.ylabel("Accuracy")

df1 = df_Friday[df_Friday['Year'] == 2018]

label_lst = df1['label'].tolist()
mean_lst = df1['mean_return'].tolist()
volatility_lst = df1['volatility'].tolist()

data = pd.DataFrame(
    {'id':[x for x in range(1, len(df1)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id', 'Label','X','Y']
)

x_values = data[['X','Y']].values
y_values = data[['Label']].values

k_range = [3,5,7,9,11]
ans = []
for k in k_range:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_values, np.ravel(y_values))
    new_instance = data[['X','Y']].values
    prediction = knn_classifier.predict(new_instance)
    correct_num = sum(prediction == np.asarray(label_lst))
    ans.append(correct_num/len(label_lst))
    print("k=",k,", The accuracy is %.2f" %(ans[-1]*100),'%',sep='')

plt.title("Accuracies of K")
plt.xlabel('the value of k')
plt.ylabel('accuracy (%)')
y_ = [round(num*100, 2) for num in ans]

plt.plot(k_range,y_)
plt.show()