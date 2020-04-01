# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
import csv
import matplotlib.pyplot as plt

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

        # set threshold [0,10%]
        bar = []
        bar_profit_avg_per_trade = []
        x = 0
        while x <= 100:
            bar.append(x / 1000)
            x += 1

        for point in range(len(bar)):
            # given 100 points of value, trade only if overnight return > bar[point]
            for i in range(1, days):
                if stock_open[i] < stock_close[i - 1] and (abs(stock_open[i] - stock_close[i-1]) / stock_close[i-1]) > bar[point]:  # SELL SHORT
                    daily_s = (100 / stock_open[i]) * (stock_open[i] - stock_close[i])
                    ds = round(daily_s, 2)
                    cnt_sell_short += 1
                    sell_short_profit += ds

                elif stock_open[i] >= stock_close[i - 1] and (abs(stock_open[i] - stock_close[i-1]) / stock_close[i-1]) > bar[point]: # LONG
                    daily_l = (100 / stock_open[i]) * (stock_close[i] - stock_open[i])
                    dl = round(daily_l, 2)
                    cnt_long += 1
                    long_profit += dl

                if i == days - 1:
                    sell_short_profit = round(sell_short_profit, 2)
                    long_profit = round(long_profit, 2)
                    # print(sell_short_profit, long_profit, cnt_sell_short, cnt_long, bar[point])
                    sum_of_trade = cnt_long + cnt_sell_short
                    sum_of_profit = sell_short_profit + long_profit

                    if sum_of_trade != 0:
                        avg_ = round(sum_of_profit / sum_of_trade, 2)
                        bar_profit_avg_per_trade.append(avg_)
                    elif sum_of_trade == 0:
                        bar_profit_avg_per_trade.append(0)

                    # initialize per round
                    sell_short_profit = 0
                    long_profit = 0
                    cnt_sell_short = 0
                    cnt_long = 0
        print()
        print("In conclusion, I find that the optimal threshold value is from 4% to 4.2%, which has $1.33 profit in "
              "average per trade.\n However, once the threshold value becomes greater than 4%, the average profit will dramatically "
              "fall down, \nsince the overnight return of LYFT can be rarely reached more than 4%, and thus we reduce the numbers of transcations,\n "
              "and as the value increases, the curve tends to be stationary at zero because nothing trade we made.")


        plt.plot(bar, bar_profit_avg_per_trade)
        plt.xlabel("Threshold Value")
        plt.ylabel("Average Profit per Trade")
        plt.title("LYFT")
        plt.show()
        #
        # print("The LONG profit:", long_profit)
        # print("The SELL SHORT profit:", sell_short_profit)
        # print("The avg profit: ", round(sell_short_profit + long_profit, 2), " %", sep='')
        # print("More profitable:", "LONG" if long_profit > sell_short_profit else "SELL SHORT")

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)












