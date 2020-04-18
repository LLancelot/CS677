import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


wd = os.getcwd()
ticker = 'LIN'
input_dir = wd
# ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)


label_lst = df['label'].tolist()
mean_lst = df['mean_return'].tolist()
volatility_lst = df['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

data1 = pd.DataFrame(
    {'week_id':[x for x in range(1, len(df)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['week_id', 'Label','X','Y']
)

x_ = data1[['X','Y']].values
label_ = data1[['Label']].values

np.random.seed(1)

kmeans_classifier = KMeans(n_clusters=4)
y_kmeans = kmeans_classifier.fit_predict(x_)
centriods = kmeans_classifier.cluster_centers_

predicted = kmeans_classifier.predict(x_)

# [green, red]
cluster_green_red = [[0,0] for _ in range (4)]

for i, cluster_id in enumerate(predicted):
    if label_lst[i] == 'green':
        cluster_green_red[cluster_id][0] += 1
    elif label_lst[i] == 'red':
        cluster_green_red[cluster_id][1] += 1

for i, each in enumerate(cluster_green_red):
    percent_green = round(100* each[0] / sum(each),2)
    percent_red = round(100* each[1] / sum(each),2)
    if percent_green > 90 or percent_red > 90:
        flag = "green weeks" if percent_green > percent_red else "red weeks"
        print("We find the cluster",i+1,"is pure since its",flag,"taking more than 90%.")
