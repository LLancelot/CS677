import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd

ticker_file = os.path.join(input_dir,ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir,ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)
df_Friday = df[df['Weekday'] == 4]

df1 = df_Friday[df_Friday['Year'] == 2018]
# df_Friday = df_Friday[df_Friday['Year'] == 2019]
df2 = df_Friday[df_Friday['Year'] == 2019]

label_lst = df1['label'].tolist()
mean_lst = df1['mean_return'].tolist()
volatility_lst = df1['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst) + 1)]

data1 = pd.DataFrame(
    {'id': [x for x in range(1, len(df1) + 1)],
     'Label': label_lst,
     'X': mean_lst,
     'Y': volatility_lst},
    columns=['id', 'Label', 'X', 'Y']
)

data2 = pd.DataFrame(
    {
        'week_id': [x for x in range(1, len(df1) + 1)],
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
    columns=['week_id', 'Label', 'X', 'Y']
)

year1_x_value = data1[['X', 'Y']].values
year1_Label = data1[['Label']].values

year2_x_value = data2[['X', 'Y']].values
year2_Label = data2[['Label']].values

fri_price_lst = df2['Adj Close'].tolist()
days_fri = len(fri_price_lst)

class Custom_KNN():
    def __init__(self, number_neighbors_k, distance_parameter_p):

        self.number_neighbors_k = number_neighbors_k
        self.distance_parameter_p = distance_parameter_p
        self.test_predict_Label = []
        self.accuracy = {1: 0, 1.5: 0, 2: 0}

    def fit(self, X, Labels):
        self.X = X
        self.Labels = Labels

    def predict(self, new_x: pd.DataFrame):
        # 1.input: test_x_value
        # 2. distance(p)
        self.fit(year1_x_value, year1_Label)
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
                    self.test_predict_Label.append('green') if cnt_green > cnt_red else self.test_predict_Label.append(
                        'red')
                    p_distance = [0] * len(df1)

        # print(sum(np.asarray(self.test_predict_Label) == np.asarray(df2['label'].tolist())))
        self.accuracy[self.distance_parameter_p] = round(
            100 * sum(np.asarray(self.test_predict_Label) == np.asarray(df2['label'].tolist())) / len(df2), 2)
        # print(self.accuracy[self.distance_parameter_p])
        res = self.test_predict_Label.copy()
        self.test_predict_Label = []
        return [self.accuracy[self.distance_parameter_p], res]

    def draw_decision_boundary(self, new_x:pd.DataFrame):
        k = (1.69316 - 1.096) / (0.113051 - 0.19522)
        b = 1.096 - k * (-0.113051)
        x1, x2 = -0.5, 0
        y1, y2 = k * x1 + b, k * x2 + b
        plt.plot([x1, x2], [y1, y2])


        # pick GREEN(id = 37), RED(id = 39)
        # frame_id = new_x['week_id'] - 1
        # year2_x_value[frame_id]
        self.fit(year1_x_value, year1_Label)
        print(new_x['week_id'].values[0])
        frame_id = new_x['week_id'].values[0] - 1

        measureDistance = [0] * len(df1)
        for idx in range(len(df1)):
            measureDistance[idx] = np.linalg.norm(year2_x_value[frame_id] - self.X[idx], ord=self.distance_parameter_p)
            if idx == len(df1) - 1:
                for dis in sorted(measureDistance)[:self.number_neighbors_k]:
                    plt.plot([self.X[measureDistance.index(dis)][0], new_x['X']],[self.X[measureDistance.index(dis)][1], new_x['Y']], ls = 'dotted')
                    plt.annotate(str(measureDistance.index(dis)),xy=(self.X[measureDistance.index(dis)][0],self.X[measureDistance.index(dis)][1]))
                measureDistance = [0] * len(df1)
        plt.title("Draw boundary")
        plt.scatter(new_x['X'],new_x['Y'],c=new_x['Label'])
        plt.annotate("week_"+str(new_x['week_id'].values) , xy=(new_x['X'].values,new_x['Y'].values))
        plt.xlabel("μ")
        plt.ylabel("σ")

    def __str__(self):
        return "number_neightbors_k = " + str(self.number_neighbors_k) + ", p = " + str(self.distance_parameter_p)

    def setP(self, para_p):
        self.distance_parameter_p = para_p

    def tradingWithLabels_and_compare(self, test_x, para_p):
        self.distance_parameter_p = para_p
        label_predict_by_p = {1:None , 1.5:None, 2:None}
        label_predict_by_p[para_p] = self.predict(new_x=test_x)[1]

        myTrading_value = 0
        hold_strategy_value = 0
        myTrading_shares = 0
        hold_strategy_shares = 0

        cur_label = 'green'

        begin = 0

        lst_daily_value = []
        lst_hold_value = []
        for i in range(days_fri - 1):
            if label_predict_by_p[para_p][i + 1] == 'green':
                myTrading_value += 100
                myTrading_shares += 100 / fri_price_lst[i]
                begin = i
                break

        for i in range(begin, days_fri - 1):
            if label_predict_by_p[para_p][i + 1] == 'green':
                if cur_label == 'green':
                    lst_daily_value.append(myTrading_value)
                    continue
                elif cur_label == 'red':
                    # we should buy all i have
                    myTrading_shares += myTrading_value / fri_price_lst[i]
                    lst_daily_value.append(myTrading_value)
                    cur_label = 'green'

            elif label_predict_by_p[para_p][i + 1] == 'red':
                if cur_label == 'red':
                    lst_daily_value.append(myTrading_value)
                    continue

                elif cur_label == 'green':
                    # sell shares
                    myTrading_value = myTrading_shares * fri_price_lst[i]
                    myTrading_shares = 0
                    lst_daily_value.append(myTrading_value)
                    cur_label = 'red'
        lst_daily_value.append(myTrading_value)

        # buy-hold strategy
        hold_strategy_shares = 100 / fri_price_lst[begin]

        for i in range(days_fri):
            lst_hold_value.append(hold_strategy_shares * fri_price_lst[i])

        return [lst_daily_value,lst_hold_value]

p_val = [1, 1.5, 2]
p_accuracy = []
test_x = data2[['X', 'Y']].values

draw_week_id = [37,39]
myKNN = Custom_KNN(5, 1)

p_trading_res1 = myKNN.tradingWithLabels_and_compare(test_x, 1)

p_trading_res2 = myKNN.tradingWithLabels_and_compare(test_x, 1.5)

p_trading_res3 = myKNN.tradingWithLabels_and_compare(test_x, 2)

x_axis = [x for x in range(0, days_fri)]
plt.plot(x_axis, p_trading_res1[0],c="blue")
plt.plot(x_axis, p_trading_res2[0],c='yellow')
plt.plot(x_axis, p_trading_res3[0],c='red')
plt.plot(x_axis, p_trading_res1[1],c='pink')

plt.legend(['p=1','p=1.5','p=2','Buy-Hold'])
plt.xlabel("Week")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Strategies Comparisons")

print("According to the graph, p=1 has the largest portfolio value"
      "\nat the end of the year.")
plt.show()