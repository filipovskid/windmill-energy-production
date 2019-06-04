# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/wpph_all.csv')
#data_arr = df.values

def correlation(df):
    for atr in range(4, 11):
        fig = plt.figure()
        test = df.groupby(['Year', 'Month'])[[df.columns[atr], 'Hourly production [kWh]']].corr().iloc[0::2,-1]
        for year in [2015, 2016, 2017]:
            plt.subplot(1, 3, year - 2014) # + (atr - 4)*3)
            plt.bar(range(1, 13), test[year], color = 
                    ['r' if abs(i) >= 0.5 else 'gray' for i in test[year]])
            plt.xlabel(f'months {year}')
    
        fig.text(0.5, 0.90, df.columns[atr], ha='center')
        fig.text(0.04, 0.5, 'Correlation', va='center', rotation='vertical')
        
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data.to_csv('data/wpph_c_train.csv', encoding='utf-8', index=False, header=False)
    test_data.to_csv('data/wpph_c_test.csv', encoding='utf-8', index=False, header=False)
    
correlation(df)