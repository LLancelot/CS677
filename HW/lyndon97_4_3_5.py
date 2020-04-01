'''
using the optimal k in 2018 to predict 2019
confusion matrix
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)

df_Friday = df[df['Weekday'] == 4]

df2 = df_Friday[df_Friday['Year'] == 2019]

label_lst = df2['label'].tolist()
mean_lst = df2['mean_return'].tolist()
volatility_lst = df2['volatility'].tolist()

data = pd.DataFrame(
    {'id':[x for x in range(1, len(df2)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id', 'Label','X','Y']
)

x_values = data[['X','Y']].values
y_values = data[['Label']].values

k = 5

knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(x_values, np.ravel(y_values))
new_instance = data[['X','Y']].values
prediction = knn_classifier.predict(new_instance)
correct_num = sum(prediction == np.asarray(label_lst))

true_lst = np.asarray(label_lst)

#
# TP = 0
# FN = 0
# FP = 0
# TN = 0
# for i in range(len(prediction)):
#     if true_lst[i] == 'green' and prediction[i] == 'green':
#         TP += 1
#     elif true_lst[i] == 'green' and prediction[i] == 'red':
#         FN += 1
#     elif true_lst[i] == 'red' and prediction[i] == 'green':
#         FP += 1
#     elif true_lst[i] == 'red' and prediction[i] == 'red':
#         TN += 1
matrix = confusion_matrix(y_true = np.asarray(label_lst), y_pred= prediction)
print(matrix)

print("The recall =",matrix[0][0] / sum(matrix[0]))
# print("The true negative rate =",matrix[1][1] / sum(matrix[1]))