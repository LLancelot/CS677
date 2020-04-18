'''
implement kmeans classifier
random initial centriods and k = 3.'
'''
import os
import pandas as pd
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

kmeans_classifier = KMeans(n_clusters=3)
y_kmeans = kmeans_classifier.fit_predict(x_)
centriods = kmeans_classifier.cluster_centers_

inertia_list = []
for k in range(1,9):
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(x_)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)

fig, ax = plt.subplots(1, figsize = (7,5))
plt.plot(range(1,9), inertia_list, marker = 'o', color = 'green')
plt.legend()
plt.xlabel('number of clusters: k')
plt.ylabel('inertia')
plt.tight_layout()

print("The best k = 4")
plt.show()