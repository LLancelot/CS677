import os
import pandas as pd
import numpy as np

wd = os.getcwd()

input_dir = wd

file_read = os.path.join(input_dir,"tips.csv")
df = pd.read_csv(file_read)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# average tip for lunch and for dinner
average_tip_lunch = np.mean(df.tip[df.time == 'Lunch']/df.total_bill[df.time == 'Lunch'])
average_tip_dinner = np.mean(df.tip[df.time == 'Dinner']/df.total_bill[df.time == 'Dinner'])

print("The average tip percentage for lunch is",str(round(average_tip_lunch*100,2))+"%")
print("The average tip percentage for dinner is",str(round(average_tip_dinner*100,2))+"%")
