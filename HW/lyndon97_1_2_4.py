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

        bar_ss_profit_avg_per_trade = []
        bar_long_profit_avg_per_trade = []

        x = 0
        while x <= 100:
            bar.append(x / 1000)
            x += 1

        for point in range(len(bar)):
            # given 100 points of value, trade only if overnight return > bar[point]
            for i in range(1, days):
                if stock_open[i] < stock_close[i - 1] and (
                        abs(stock_open[i] - stock_close[i - 1]) / stock_close[i - 1]) > bar[point]:  # SELL SHORT
                    daily_s = (100 / stock_open[i]) * (stock_open[i] - stock_close[i])
                    ds = round(daily_s, 2)
                    cnt_sell_short += 1
                    sell_short_profit += ds

                elif stock_open[i] >= stock_close[i - 1] and (
                        abs(stock_open[i] - stock_close[i - 1]) / stock_close[i - 1]) > bar[point]:  # LONG
                    daily_l = (100 / stock_open[i]) * (stock_close[i] - stock_open[i])
                    dl = round(daily_l, 2)
                    cnt_long += 1
                    long_profit += dl

                if i == days - 1:
                    sell_short_profit = round(sell_short_profit, 2)
                    long_profit = round(long_profit, 2)

                    if cnt_sell_short != 0:
                        avg_ss = round(sell_short_profit / cnt_sell_short, 2)
                        bar_ss_profit_avg_per_trade.append(avg_ss)

                    if cnt_sell_short == 0:
                        bar_ss_profit_avg_per_trade.append(0)

                    if cnt_long != 0:
                        avg_long = round(long_profit / cnt_long, 2)
                        bar_long_profit_avg_per_trade.append(avg_long)

                    if cnt_long == 0:
                        bar_long_profit_avg_per_trade.append(0)

                    # initialize per round (every 209 times)
                    sell_short_profit = 0
                    long_profit = 0
                    cnt_sell_short = 0
                    cnt_long = 0

        #
        # print("The LONG profit:", long_profit)
        # print("The SELL SHORT profit:", sell_short_profit)
        # print("The avg profit: ", round(sell_short_profit + long_profit, 2), " %", sep='')
        # print("More profitable:", "LONG" if long_profit > sell_short_profit else "SELL SHORT")

        print("My Conclusion:")
        print(
            "As the graph is shown, it is obviously that the short position performs much better than long position,\n since it is more profitable. "
            "In particularly, when we look at the short strategy, we can find the optimal threshold value is approximately at 4%, \n"
            "which gives us the most average profit per trade, that is $7.8. \nUnfortunately, the long strategy is worse and unsuitable for this stock, "
            "because the profits we get are always negative.")

        plt.title('Average Profit of Long and Short Strategies (LYFT)')
        plt.plot(bar, bar_ss_profit_avg_per_trade)
        plt.plot(bar, bar_long_profit_avg_per_trade)
        plt.legend(['Short', 'Long'])
        plt.xlabel("Threshold Value")
        plt.ylabel("Average Profit per Trade")
        plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)
