
import os
import pandas as pd
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

df = pd.read_csv(ticker_file)
df2018 = df[df['Year'] == 2018]
# df2019 = df[df['Year'] == 2019]
def bollinger(k, w:int):
    profit = 0
    buy_cost = 0
    ss_cost = 0
    buy_share = 0
    ss_share = 0
    pos = "no"
    close_time = 0
    '''
    row['adj close'] : closing price
    '''
    for index, row in df2018.iterrows():
        # daily MA

        MA = df2018[index:index+w]['Adj Close'].mean()
        # print(MA, type(MA),row['volatility'])
        upper_bound = MA + k*row['volatility']
        lower_bound = MA - k*row['volatility']
        # print(index, row['Adj Close'], lower_bound , upper_bound, pos, buy_cost,buy_share, ss_cost, ss_share, "profit:", profit)
        if row['Adj Close'] >= lower_bound and row['Adj Close'] <= upper_bound:
            continue

        elif row['Adj Close'] < lower_bound: # LONG
            if pos == 'no':
                buy_cost += 100
                buy_share += 100 / row['Adj Close']
                pos = "long"
            elif pos == 'short':
                profit += ss_cost - ss_share * row['Adj Close']
                ss_cost = 0
                ss_share = 0
                close_time += 1
                pos = "no"
            elif pos == 'long':
                continue

        elif row['Adj Close'] > upper_bound: #SHORT
            if pos == 'no':
                ss_cost += 100
                ss_share += 100 / row['Adj Close']
                pos = "short"
            elif pos == 'long':
                profit += buy_share * row['Adj Close'] - buy_cost
                buy_share = 0
                buy_cost = 0
                pos = 'no'
                close_time += 1
            elif pos == 'short':
                continue
    avg_profit_perTrans = profit / close_time
    return avg_profit_perTrans
        # print(upper_bound)


'''
(k,w:days)
'''
k_value = [0.5, 1, 1.5, 2, 2.5]
for num in k_value:
    for day in range(10,52):
        print("k=",num,"w=",day,"Avg Profit:",bollinger(num,day))
# print(Bollinger(0.5, 10))

scatter_positive = pd.DataFrame(columns=('w','k','value'))
scatter_negative = pd.DataFrame(columns=('w','k','value'))
row_positive = 0
row_negative = 0

for w in range(10,52):
    for k in k_value:
        a = bollinger(k,w)
        if a > 0:
            scatter_positive.loc[row_positive] = {'w':w,'k':k,'value':a}
            row_positive += 1
        elif a < 0:
            scatter_negative.loc[row_negative] = {'w':w,'k':k,'value':a}
            row_negative += 1

plt.figure(figsize = (10,5))
plt.title('2018 Average P/L per transaction with change of W and k')

# plt.title('2019 Average P/L per transaction with change of W and k')

plt.xlabel('W', fontsize = 14)
plt.ylabel('k', fontsize = 14)
plt.scatter(scatter_positive['w'],scatter_positive['k'],s=scatter_positive['value']*20,c='green')
plt.scatter(scatter_negative['w'],scatter_negative['k'],s=-scatter_negative['value']*20,c='red')
plt.legend(['avg_profit','loss'])
plt.show()
