import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance



init_centriods = [[0.57279643, 1.46545772],
            [-0.50679559, 1.26670541],
            [0.26310469, 0.63515471],
            [0.08027083, 2.94823033]]

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

'''
1. for each point calculate (c[0~3], x_[i], p = 1 / 1.5 / 2)
2. record result for each loop
3. check until there is no change in membership 
'''

def episode(points, centriods, p):
    result_cluster = [[] for _ in range(4)]
    # calculate Kmeans by minkowski with p
    for i, point in enumerate(points):
        min_distance = 2*31
        belongs = 0
        for j, centriod in enumerate(centriods):
            cur_distance = distance.minkowski(point, centriod, p)
            if cur_distance < min_distance:
                min_distance = cur_distance
                belongs = j
        result_cluster[belongs].append(i)
    return result_cluster

def calculate_centriod(res_cluster):
    new_centriods = [[0, 0] for _ in range(4)]
    for i, cls in enumerate(res_cluster):
        sum_X, sum_Y = 0, 0
        for _id in cls:
            sum_X += x_[_id][0]
            sum_Y += x_[_id][1]
        mean_X = sum_X / len(cls)
        mean_Y = sum_Y / len(cls)
        new_centriods[i][0], new_centriods[i][1] = mean_X, mean_Y
    return new_centriods

def manual_KMeans(p):
    points = x_
    centroids = init_centriods
    result_ = None
    SSE = 0
    new_cluster = episode(points, centroids, p)
    new_centriods = None
    while result_ != new_cluster:
        # find until no change in each clusters
        result_ = new_cluster
        new_centriods = calculate_centriod(new_cluster)
        new_cluster = episode(points, new_centriods, p)

    final_cluster = new_cluster
    final_centriods = new_centriods
    # got final cluster and calculate SSE

    for i, cls in enumerate(final_cluster):
        for _id in cls:
            SSE += distance.minkowski(points[_id], final_centriods[i], 2)

    return round(SSE,2)

print("Summary of manual KMeans writen by myself:")
print("\nThe SSE of Euclidean distance is", manual_KMeans(2))
print("The SSE of Minkowski distance (p = 1.5) is", manual_KMeans(1.5))
print("The SSE of Minkowski distance (p = 1) is", manual_KMeans(1))
print("\nTherefore, using Euclidean distance or Minkowski distance (p = 1.5) is best in KMeans,"
      "\nsince its SSE is the least.")
