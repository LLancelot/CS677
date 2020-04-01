import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_16_19_new.csv')
plot_dir = os.path.join(input_dir)

try:
    df = pd.read_csv(ticker_file)
    df['Return'] = 100.0 * df['Return']


    pos = len(df[df['Return'] > 0])
    neg = len(df[df['Return'] < 0])

    ticker_mean = df['Return'].mean()
    ticker_std = df['Return'].std()

    # print(ticker_mean - 2*ticker_std, ticker_mean+2*ticker_std)
    interval_low = ticker_mean - 2*ticker_std
    interval_high = ticker_mean + 2*ticker_std

    days_less_2SD = len(df[df['Return'] < interval_low])
    days_more_2SD = len(df[df['Return'] > interval_high])

    print("From 2016 to 2019")
    print("The number of days that daily returns below μ-2σ:", days_less_2SD)
    print("The number of days that daily returns above μ+2σ:", days_more_2SD)

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

