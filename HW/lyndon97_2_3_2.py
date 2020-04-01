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

    less_mean = len(df[df['Return'] < ticker_mean])
    greater_mean = len(df[df['Return'] > ticker_mean])
    print("The number of days positive:", pos)
    print("The number of days negative:", neg)

    days = len(df['Return'])
    percent_less_mean = less_mean / days
    percent_greater_mean = greater_mean / days
    print("The percentage of days with return less than mean: " + "{:.2%}".format(percent_less_mean))
    print("The percentage of days with return greater than mean: " + "{:.2%}".format(percent_greater_mean))
    print("My conclusion: Through 4 years, positive days is more than negative days")



    df2016 = df[df['Year'] == 2016]
    df2017 = df[df['Year'] == 2017]
    df2018 = df[df['Year'] == 2018]
    df2019 = df[df['Year'] == 2019]

    pos16 = len(df2016[df2016['Return'] > 0])
    neg16 = len(df2016[df2016['Return'] < 0])
    pos17 = len(df2017[df2017['Return'] > 0])
    neg17 = len(df2017[df2017['Return'] < 0])
    pos18 = len(df2018[df2018['Return'] > 0])
    neg18 = len(df2018[df2018['Return'] < 0])
    pos19 = len(df2019[df2019['Return'] > 0])
    neg19 = len(df2019[df2019['Return'] < 0])

    mean16 = df2016['Return'].mean()
    mean17 = df2017['Return'].mean()
    mean18 = df2018['Return'].mean()
    mean19 = df2019['Return'].mean()

    less_mean_16 = len(df2016[df2016['Return'] < mean16])
    less_mean_17 = len(df2017[df2017['Return'] < mean17])
    less_mean_18 = len(df2018[df2018['Return'] < mean18])
    less_mean_19 = len(df2019[df2019['Return'] < mean19])

    greater_mean_16 = len(df2016[df2016['Return'] > mean16])
    greater_mean_17 = len(df2017[df2017['Return'] > mean17])
    greater_mean_18 = len(df2018[df2018['Return'] > mean18])
    greater_mean_19 = len(df2019[df2019['Return'] > mean19])

    percent_less_mean_16 = less_mean_16 / len(df2016['Return'])
    percent_less_mean_17 = less_mean_17 / len(df2017['Return'])
    percent_less_mean_18 = less_mean_18 / len(df2018['Return'])
    percent_less_mean_19 = less_mean_19 / len(df2019['Return'])

    percent_greater_mean_16 = greater_mean_16 / len(df2016['Return'])
    percent_greater_mean_17 = greater_mean_17 / len(df2017['Return'])
    percent_greater_mean_18 = greater_mean_18 / len(df2018['Return'])
    percent_greater_mean_19 = greater_mean_19 / len(df2019['Return'])

    data = {'year':['2016-2019','2016','2017','2018','2019'],
            'trading days':[days,len(df2016['Return']),len(df2017['Return']),len(df2018['Return']),len(df2019['Return'])],
        'μ':[ticker_mean,mean16,mean17,mean18,mean19],'%days < μ':
            ["{:.2%}".format(percent_less_mean),"{:.2%}".format(percent_less_mean_16),"{:.2%}".format(percent_less_mean_17),
             "{:.2%}".format(percent_less_mean_18),"{:.2%}".format(percent_less_mean_19)],
            '%days > μ':["{:.2%}".format(percent_greater_mean),"{:.2%}".format(percent_greater_mean_16),"{:.2%}".format(percent_greater_mean_17),
                         "{:.2%}".format(percent_greater_mean_18),
                         "{:.2%}".format(percent_greater_mean_19)]}
    newdf = pd.DataFrame(data)
    print()
    print("*********** Result Table ***********")
    print(newdf)

    year1 = '2016'
    year2 = '2019'
    start_date = year1 + '-01-01'
    end_date = year2 + '-12-31'
    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    low_return = -5
    high_return = 5
    df = df[(df['Return'] > low_return) & (df['Return'] < high_return)]
    fig = plt.figure()
    returns_list = df['Return'].values
    plt.hist(returns_list, density=True, bins=30, label='Daily Returns')
    x = np.linspace(low_return, high_return, 1000)
    # pos = len(df[df['Return'] > 0])
    # neg = len(df[df['Return'] < 0])
    # zero = len(df[df['Return'] == 0.0])
    # ticker_mean = df['Return'].mean()
    # ticker_std = df['Return'].std()

    # plt.plot(x, norm.pdf(x, ticker_mean, ticker_std), color='red',
    #          label='Normal, ' + r'$\mu=$' + str(round(ticker_mean, 2)) +
    #                ', ' + r'$\sigma=$' + str(round(ticker_std, 2)))
    # # plt.title('daily returns for ' + ticker + ' for year ' + year1)
    # plt.legend()
    # output_file = os.path.join(plot_dir, 'returns_' + year1 + '_' + ticker + '_' + str(year2) + '.png')
    # plt.savefig(output_file)
    # plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

