import os
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
output_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)
df_Friday = df[df['Weekday'] == 4]

# df_Friday = df_Friday[df_Friday['Year'] == 2018]
df_Friday = df_Friday[df_Friday['Year'] == 2019]

label_lst = df_Friday['label'].tolist()
mean_lst = df_Friday['mean_return'].tolist()
volatility_lst = df_Friday['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

removed_dot = [4,17,20,25,32]

k = (1.69316-1.096)/(0.113051-0.19522)
b = 1.096 - k*(-0.113051)
x1,x2=-0.5,0
y1,y2 = k*x1+b,k*x2+b
plt.plot([x1,x2],[y1,y2])

correct_green = 0
red_in_green = 0
correct_red = 0
green_in_red = 0
plt.title("Predict Year 2 (2019):")
for i in range(len(label_lst)):
    if mean_lst[i]*k + b < volatility_lst[i]:
        # predict green, check if is correct.
        if label_lst[i] == 'green':
            correct_green += 1
        else:
            red_in_green += 1
        plt.scatter(mean_lst[i],volatility_lst[i],c = 'green')
    elif mean_lst[i]*k + b > volatility_lst[i]:
        if label_lst[i] == 'red':
            correct_red += 1
        else:
            green_in_red += 1
        plt.scatter(mean_lst[i],volatility_lst[i],c = 'red')

print(correct_green,green_in_red,correct_red,red_in_green)
# for xi, yi, wi, label in zip(mean_lst,volatility_lst,week_id,label_lst):
#     if label == 'green':
#         plt.scatter(xi,yi,c='green')
#         plt.annotate(str(wi), xy=(xi,yi))
#     elif label == 'red':
#         plt.scatter(xi, yi, c='red')
#         # plt.annotate(str(wi), xy=(xi, yi))

# plt.plot([1,0],[0,1],color='black',ls='dotted')



# get the slope of line

plt.show()



