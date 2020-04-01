import os
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)
df_Friday = df[df['Weekday'] == 4]

label_lst = df_Friday['label'].tolist()
mean_lst = df_Friday['mean_return'].tolist()
volatility_lst = df_Friday['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

plt.title("Examine the Label based on μ and σ")
for i in range(len(label_lst)):
    if label_lst[i] == 'green':
        plt.scatter(mean_lst[i],volatility_lst[i],c = 'green')
    elif label_lst[i] == 'red':
        plt.scatter(mean_lst[i],volatility_lst[i],c = 'red')

# for xi, yi, wi, label in zip(mean_lst,volatility_lst,week_id,label_lst):
#     if label == 'green':
#         plt.scatter(xi,yi,c='green')
#         plt.annotate(str(wi), xy=(xi,yi))
#     elif label == 'red':
#         plt.scatter(xi, yi, c='red')
#         plt.annotate(str(wi), xy=(xi, yi))

print("The patterns of the plot is not quite obvious, generally the "
      "boundary\n of red and green dots is around μ = 0, and yes,"
      "for higher σ and µ, \nthere are more green points than red.")

plt.xlabel("µ: Average of daily returns")
plt.ylabel("σ: Volatility")
plt.show()



