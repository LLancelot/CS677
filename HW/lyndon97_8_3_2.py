'''
implement a Gaussian NB classifier
'''
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix


wd = os.getcwd()
ticker = 'LIN'
input_dir = wd
# ticker_file = os.path.join(input_dir, ticker + '_18_19_label_copy.csv')
ticker_file = os.path.join(input_dir, ticker + '_18_19_label_Friday.csv')
df = pd.read_csv(ticker_file)

df1 = df[df['Year'] == 2018]
df2 = df[df['Year'] == 2019]

df1_green = df1[df1["label"] == "green"]
df1_red =  df1[df1["label"] == "red"]

mu_df1_green_mean = df1_green['mean_return'].values.mean()
mu_df1_green_std = df1_green['mean_return'].values.std()
sgm_df1_green_mean = df1_green['volatility'].values.mean()
sgm_df1_green_std = df1_green['volatility'].values.std()

mu_df1_red_mean = df1_red['mean_return'].values.mean()
mu_df1_red_std = df1_red['mean_return'].values.std()
sgm_df1_red_mean = df1_red['volatility'].values.mean()
sgm_df1_red_std = df1_red['volatility'].values.std()

p_green = len(df1_green) / len(df1)
p_red = len(df1_red)/ len(df1)

predicted = []

data2 = pd.DataFrame(
    {
        'week_id':[x for x in range(1, len(df2)+1)],
        'Label': df2['label'].tolist(),
        'X': df2['mean_return'].tolist(),
        'Y': df2['volatility'].tolist()},
        columns = ['week_id','Label','X','Y']
)

for index, row in df2.iterrows():
    mu, sigma = row['mean_return'], row['volatility']
    prob_mu_green = norm.pdf((mu - mu_df1_green_mean) / mu_df1_green_std)
    prob_sigma_green = norm.pdf((sigma - sgm_df1_green_mean) / sgm_df1_green_std)
    prob_mu_red = norm.pdf((mu - mu_df1_red_mean) / mu_df1_red_std)
    prob_sigma_red = norm.pdf((sigma - sgm_df1_red_mean)/sgm_df1_red_std)

    posterior_red = p_red*prob_mu_red*prob_sigma_red
    posterior_green = p_green*prob_mu_green*prob_sigma_green
    normalized_red = posterior_red / (posterior_red+posterior_green)
    normalized_green = posterior_green / (posterior_green+posterior_red)

    predicted.append("red") if normalized_red > normalized_green else predicted.append("green")

matrix = confusion_matrix(y_true=data2[['Label']].values.ravel(),
                          y_pred=np.array(predicted))
print("Confusion Matrix:")
print(matrix)
