# best w = 26 in year 2018.

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

df = pd.read_csv(ticker_file)
df2019 = df[df['Year'] == 2019]

dailyPrice_2019 = df2019['Adj Close'].tolist()

data2 = pd.DataFrame({
    'id_day':[x for x in range(1, len(df2019)+1)],
    'close_price': dailyPrice_2019},
    columns = ['id_day','close_price']
)

w = 26
r2_lst = []
for index, row in data2.iterrows():
    if index + w == len(data2):
        break

    model = LinearRegression()
    X = data2[index:index + w]['id_day'].values
    y = data2[index:index + w]['close_price'].values
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)
    model.fit(X, y)
    predict_price = model.predict([X[-1] + 1])

    r2 = round((predict_price[0][0] - data2['close_price'][index+w])**2,2)
    r2_lst.append(r2)

print("The average r**2 value is",round(sum(r2_lst)/len(r2_lst),2))
print()
print("As can be seen from the figure, the data fluctuated greatly for about 10 days, "
      "\nindicating that the predicted value obtained by linear regression and "
      "\nthe actual value deviated in a stable range in most other trading days.")

plt.title("r**2 value with w = 26 in Year 2019 Prediction")
plt.plot([x for x in range(27,len(data2)+1)],r2_lst,color="red")
plt.xlabel("Days")
plt.ylabel("r**2 value")
plt.show()




