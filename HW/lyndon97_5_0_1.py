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

df_Friday = df_Friday[df_Friday['Year'] == 2018]
# df_Friday = df_Friday[df_Friday['Year'] == 2019]

label_lst = df_Friday['label'].tolist()
mean_lst = df_Friday['mean_return'].tolist()
volatility_lst = df_Friday['volatility'].tolist()
week_id = [x for x in range(1, len(label_lst)+1)]

removed_dot = [4,17,20,25,32]

plt.title("Year 1 (2018):")
for i in range(len(label_lst)):
    if i+1 not in removed_dot:
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
#         # plt.annotate(str(wi), xy=(xi, yi))

plt.plot([-0.19522,-0.113051],[1.69316,1.096],color='black',ls='dotted')
# plt.plot([1,0],[0,1],color='black',ls='dotted')



# get the slope of line
k = (1.69316-1.096)/(0.113051-0.19522)
b = 1.096 - k*(-0.113051)
x1,x2=-0.5,0
y1,y2 = k*x1+b,k*x2+b
plt.plot([x1,x2],[y1,y2])
print("The line function:","y = -7.267x + 0.274")
plt.show()

print("The patterns of year1 is different from year2, based on the plot.")



