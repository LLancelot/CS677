'''
Implement a linear and quadratic discriminant classifier. As
before, for each classifier use year 1 labels as training set and
predict year 2 labels. For each week, your feature set is (µ; σ)
for that week. Use your labels (you will have 52 weekly labels
per year) from year 1 to train your classifier and predict labels
for year 2.

'''

import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

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
lda_classifier = LDA(n_components=1)
lda_classifier.fit(year1_re_and_vo, year1_Label)

predicted_lda = lda_classifier.predict(testing_data)

b1, b2, b0 = lda_classifier.coef_[0][0], lda_classifier.coef_[0][1], lda_classifier.intercept_[0]
b1, b2, b0 = round(b1,2), round(b2, 2), round(b0, 2)
res = "y = "+str(b1)+"x1 "+str(b2)+"x2 +"+str(b0)

qda_classifier = QDA()
qda_classifier.fit(year1_re_and_vo, year1_Label)
predicted_qda = qda_classifier.predict(testing_data)

accuracy_lda = round(sum(predicted_lda==data2['Label'].values.ravel())/len(data2['Label'].values.ravel())*100,2)
accuracy_qda = round(sum(predicted_qda==data2['Label'].values.ravel())/len(data2['Label'].values.ravel())*100,2)

print("Accuracy of LDA for year 2:", str(accuracy_lda)+"%")
print("Accuracy of QDA for year 2:", str(accuracy_qda)+"%")

if accuracy_lda > accuracy_qda:
    print("The LDA classifier is better.")
else:
    print("The QDA classifier is better.")
