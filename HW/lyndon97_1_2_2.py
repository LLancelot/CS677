# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
import csv

ticker = 'LYFT'
input_dir = r'C:\Users\Lin\PycharmProjects\CS677\HW\dataset'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    print('opened file for ticker: ', ticker)
    stock_open = []
    stock_close = []
    with open(ticker_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for lines in reader:
            stock_open.append(lines[1])
            stock_close.append(lines[5])

        # remove the title 'open' & 'adj close'
        # length = 210
        stock_open = stock_open[1:]
        stock_close = stock_close[1:]

        days = len(stock_open)
        for i in range(days):
            stock_open[i] = float(stock_open[i])
            stock_close[i] = float(stock_close[i])

        # print("open:", stock_open)
        # print("close:", stock_close)
        # trading start at the second day and decide LONG or SELL SHORT
        # if open[i+1] > close[i], LONG
        # else, SELL SHORT

        sell_short_profit = 0
        long_profit = 0

        cnt_sell_short = 0
        cnt_long = 0

        for i in range(1, days):
            if stock_open[i] < stock_close[i - 1]:  # SELL SHORT
                daily_s = (100 / stock_open[i]) * (stock_open[i] - stock_close[i])
                ds = round(daily_s, 2)
                sell_short_profit += ds

            else:
                daily_l = (100 / stock_open[i]) * (stock_close[i] - stock_open[i])
                dl = round(daily_l, 2)
                long_profit += dl

        sell_short_profit = round(sell_short_profit, 2)
        long_profit = round(long_profit, 2)

        print("The LONG profit:", long_profit)
        print("The SELL SHORT profit:", sell_short_profit)
        # print("The avg profit: ", round(sell_short_profit + long_profit, 2), " %", sep='')
        print("More profitable:", "LONG" if long_profit > sell_short_profit else "SELL SHORT")

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)












