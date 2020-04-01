'''
adding the result of trading using knn
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

df_Friday = df_Friday[df_Friday['Year'] == 2019]










# using the result of knn to trading in year2
label_lst = prediction
fri_price_lst = df_Friday['Adj Close'].tolist()

myTrading_value = 0
myTrading_shares = 0

weekly_shares = []

cur_label = 'green'
days_fri = len(fri_price_lst)
begin = 0

lst_daily_value = []

for i in range(days_fri-1):
    if label_lst[i+1] == 'green':
        myTrading_value += 100
        myTrading_shares += 100 / fri_price_lst[i]
        begin = i
        weekly_shares.append(myTrading_shares)
        break

for i in range(begin, days_fri-1):
    if label_lst[i+1] == 'green':
        if cur_label == 'green':
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            continue
        elif cur_label == 'red':
            # we should buy all i have
            myTrading_shares += myTrading_value / fri_price_lst[i]
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            cur_label = 'green'

    elif label_lst[i+1] == 'red':
        if cur_label == 'red':
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            continue

        elif cur_label == 'green':
            # sell shares
            myTrading_value = myTrading_shares * fri_price_lst[i]
            myTrading_shares = 0
            lst_daily_value.append(myTrading_value)
            weekly_shares.append(myTrading_shares)
            cur_label = 'red'
lst_daily_value.append(myTrading_value)

print("Analyze the trading result based on the predicted labels in 2019,"
      "\nusing KNN (k=5),"
      "\nthe final account balance is $",round(myTrading_value,2),", start with $100.",sep='')
x_axis = [x for x in range(0, days_fri)]
plt.title("Growth of My Account")
plt.plot(x_axis, lst_daily_value)

plt.legend(['Label-based'])
plt.xlabel("Week")
plt.ylabel("Account Balance")
plt.show()