
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

df1 = df_Friday[df_Friday['Year'] == 2018]
# df_Friday = df_Friday[df_Friday['Year'] == 2019]
df2 = df_Friday[df_Friday['Year'] == 2019]

label_lst = df1['label'].tolist()
mean_lst = df1['mean_return'].tolist()
volatility_lst = df1['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

data1 = pd.DataFrame(
    {'id':[x for x in range(1, len(df1)+1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns = ['id', 'Label','X','Y']
)

data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df1)+1)],
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Label','X','Y']
)


year1_x_value = data1[['X','Y']].values
year1_Label = data1[['Label']].values



class Custom_KNN():
    def __init__(self, number_neighbors_k, distance_parameter_p):

        self.number_neighbors_k = number_neighbors_k
        self.distance_parameter_p = distance_parameter_p
        self.test_predict_Label = []
        self.accuracy = {1:0, 1.5:0, 2:0}

    def fit(self, X, Labels):
        self.X = X
        self.Labels = Labels

    def predict(self, new_x:pd.DataFrame):
        # 1.input: test_x_value
        # 2. distance(p)
        self.fit(year1_x_value,year1_Label)
        p_distance = [0] * len(df1)
        for i in range(len(new_x)):
            # each row in y2
            for j in range(len(df1)):
                p_distance[j] = np.linalg.norm(new_x[i] - self.X[j], ord=self.distance_parameter_p)
                if j == len(df1) - 1:
                    cnt_green = 0
                    cnt_red = 0
                    for dis in sorted(p_distance)[:self.number_neighbors_k]:
                        if self.Labels[p_distance.index(dis)] == 'green':
                            cnt_green += 1
                        else:
                            cnt_red += 1
                    self.test_predict_Label.append('green') if cnt_green > cnt_red else self.test_predict_Label.append('red')
                    p_distance = [0] * len(df1)

        # print(sum(np.asarray(self.test_predict_Label) == np.asarray(df2['label'].tolist())))
        self.accuracy[self.distance_parameter_p] = round(100*sum(np.asarray(self.test_predict_Label) == np.asarray(df1['label'].tolist())) / len(df1) , 2)
        # print(self.accuracy[self.distance_parameter_p])
        self.test_predict_Label = []
        return self.accuracy[self.distance_parameter_p]
    def draw_decision_boundary(self, new_x):
        pass
    def __str__(self):
        return "number_neightbors_k = " + str(self.number_neighbors_k) + ", p = " + str(self.distance_parameter_p)
    def setP(self, para_p):
        self.distance_parameter_p = para_p

p_val = [1,1.5,2]
p_accuracy = []

test_x = data1[['X','Y']].values

myKNN = Custom_KNN(5,1)

for p in p_val:
    myKNN.setP(p)
    p_accuracy.append(myKNN.predict(test_x))
    print("p=",p,", accuracy=",p_accuracy[-1],"%")

plt.title("The Relationship between p and accuracy in year1")
plt.plot(p_val,p_accuracy)
plt.xlabel("The range of p")
plt.ylabel("Accuracy(%)")

plt.show()