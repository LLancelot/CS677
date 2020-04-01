# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:35:25 2020

@author: epinsky
"""

import os
import pandas as pd
import numpy as np

# change with a file name 
file_name = os.path.join(r'C:\Users','epinsky','bu','python','data_science_with_Python','datasets','tips.csv')
df = pd.read_csv(file_name)

df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# average tip for lunch and for dinner
average_tip_lunch = np.mean(df.tip[df.time == 'Lunch']/df.total_bill[df.time == 'Lunch'])
average_tip_dinner = np.mean(df.tip[df.time == 'Dinner']/df.total_bill[df.time == 'Dinner'])

if average_tip_lunch > average_tip_dinner:
    print("Tips are higher during lunch.")
elif average_tip_lunch < average_tip_dinner:
    print("Tips are higher during dinner.")
else:
    print("Tips are equal during lunch and dinner")

# the answer was: higher during lunch