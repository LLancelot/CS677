import os
import pandas as pd
import numpy as np
from scipy.stats import f as fisher_f
from sklearn.linear_model import LinearRegression

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
# output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df_Friday = pd.read_csv(ticker_file)
# df_Friday = df[df['Weekday'] == 4]

df1 = df_Friday[df_Friday['Year'] == 2018]
df2 = df_Friday[df_Friday['Year'] == 2019]

def calculate(year,month):
    # data: month rows
    # N: length(data)
    # data_L1: part1 of data,
    global data
    if year == 2018:
        data = df1[df1['Month'] == month]
    elif year == 2019:
        data = df2[df2['Month'] == month]
    N = len(data)
    month_adj_close = data.copy()['Adj Close'].tolist()
    month_SSE = 0
    month_reg = LinearRegression().fit(np.asarray([day for day in range(1,N+1)]).reshape(-1,1), np.asarray(month_adj_close).reshape(-1,1))
    for price in month_adj_close:
        month_SSE += (price - month_reg.predict([[month_adj_close.index(price)+1]]))**2

    res_SSE = []
    for k in range(1,N):
        data_L1 = data.copy()[:k]
        data_L2 = data.copy()[k:N]
        L1_adj_close = data_L1['Adj Close'].tolist()
        L2_adj_close = data_L2['Adj Close'].tolist()

        L1_reg = LinearRegression().fit(np.asarray([day for day in range(1,k+1)]).reshape(-1,1), np.asarray(L1_adj_close).reshape(-1,1))
        L2_reg = LinearRegression().fit(np.asarray([day for day in range(k+1, N+1)]).reshape(-1,1), np.asarray(L2_adj_close).reshape(-1,1))

        L1_SSE, L2_SSE, all_SSE = 0, 0, 0
        for price in L1_adj_close:
            L1_SSE += (price - L1_reg.predict([[L1_adj_close.index(price) + 1]]))**2
        for price in L2_adj_close:
            L2_SSE += (price - L2_reg.predict([[L2_adj_close.index(price)+ k+1]]))**2
        all_SSE = L1_SSE + L2_SSE
        res_SSE.append((k,month_SSE[0][0],all_SSE[0][0],N))
    return sorted(res_SSE, key= lambda x: x[2])[0]

# map: (k, L, (L1+L2))
minK_2018 = {}
minK_2019 = {}
for m in range(1,13):
    minK_2018[m] = calculate(2018,m)

for m in range(1,13):
    minK_2019[m] = calculate(2019,m)

p_value_2018 = {}
p_value_2019 = {}

def get_p_value(year, month):
    global info
    if year == 2018:
        info = minK_2018
        f_stat = ((info[month][1] - info[month][2]) / 2) / (info[month][2] / (info[month][3] - 4))
        p_value_2018[month] = 1- fisher_f.cdf(f_stat, 2, info[month][3] - 4)
        # p_value_2018[month] = f_stat
    elif year == 2019:
        info = minK_2019
        f_stat = ((info[month][1] - info[month][2]) / 2) / (info[month][2] / (info[month][3] - 4))
        p_value_2019[month] = 1- fisher_f.cdf(f_stat, 2, info[month][3] - 4)
        # p_value_2019[month] = f_stat

for m in range(1,13):
    get_p_value(2018, m)
    get_p_value(2019, m)

for k,v in p_value_2018.items():
    if v < 0.1:
        print("In 2018","month",k,", there had significant change of price trend.")
    else:
        print("In 2018", "month", k, ", there had not significant change of price trend.")

for k,v in p_value_2019.items():
    if v < 0.1:
        print("In 2019","month",k,", there had significant change of price trend.")
    else:
        print("In 2019", "month", k, ", there had not significant change of price trend.")




