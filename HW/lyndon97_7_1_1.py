import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

df = pd.read_csv(ticker_file)
df2018 = df[df['Year'] == 2018]

dailyPrice_2018 = df2018['Adj Close'].tolist()

data1 = pd.DataFrame({
    'id_day':[x for x in range(1, len(df2018)+1)],
    'close_price': dailyPrice_2018},
    columns = ['id_day','close_price']
)

def trading_2018(w):
    profit = 0
    buy_cost = 0
    ss_cost = 0
    buy_share = 0
    ss_share = 0
    pos = "no"
    close_time = 0

    for index, row in data1.iterrows():
        if index + w == len(data1):
            return profit / close_time

        model = LinearRegression()
        X = data1[index:index+w]['id_day'].values
        y = data1[index:index+w]['close_price'].values
        X = X.reshape(len(X),1)
        y = y.reshape(len(y),1)
        model.fit(X,y)
        predict_price = model.predict([X[-1]+1])
        if predict_price > y[-1]: # LONG
            if pos == 'no':
                buy_cost += 100
                buy_share += 100 / y[-1]
                pos = 'long'
            elif pos == 'short':
                profit += ss_cost - ss_share * y[-1]
                ss_cost, ss_share = 0, 0
                pos = 'no'
                close_time +=1
            elif pos == 'long':
                continue

        elif predict_price < y[-1]: # SHORT
            if pos == 'no':
                ss_cost += 100
                ss_share += 100 / y[-1]
                pos = 'short'
            elif pos == 'long':
                profit += buy_share * y[-1] - buy_cost
                buy_share, buy_cost = 0, 0
                pos = 'no'
                close_time += 1
            elif pos == 'short':
                continue

        # last day

res = []
for w in range(5,31):
    res.append(round(trading_2018(w).tolist()[0],2))

print("The best W is 26.")

plt.title("W and avg P/L per trade")
plt.plot([x for x in range(5,31)],res,color='red')
plt.xlabel("The range of W")
plt.ylabel("Average P/L per trade ($)")
plt.show()




