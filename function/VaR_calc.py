from numpy.core.fromnumeric import mean, shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf


def mcVaR(Return, alpha=5):
    if isinstance(Return, pd.Series):
        return np.percentile(Return, alpha)
    else:
        raise TypeError("Expected format of data is pandas")

# import data
def get_data(stocks, begin, end):
    stock_data = yf.download(stocks, begin, end)
    stock_data_close = stock_data['Adj Close']
    Return = stock_data_close.pct_change()
    Return_mean = Return.mean()
    cov_Matrix = Return.cov()
    return Return_mean, cov_Matrix 

list_of_stocks = ['AAPL', 'GOOG', 'MSFT'] #Apple, Google, Microsoft
#list_of_stocks = ['KO', 'GOOG', 'GM'] #Coca-cola, Google, General Motors
stocks = [stock for stock in list_of_stocks]
end_date = dt.date(2019, 9, 25)
begin_date = end_date - dt.timedelta(days=300) #no of days for which price is observed

mean_return, cov_Mat = get_data(stocks, begin_date, end_date)

#print(cov_Mat)
#weight factor for each stock inside portfolio
#what percentage of portfolio does every stock have inside
#sum of these values must be equal to 1
np.random.seed(10)
weight_of_stocks = np.random.random(len(mean_return)) 
weight_of_stocks /= np.sum(weight_of_stocks)

# Monte Carlo metod
mc_sim = 5000 # no of simulations
T = 100 #observed time period

meanM = np.full(shape=(T, len(weight_of_stocks)), fill_value = mean_return)
meanM = meanM.T

portfolio_no_sim = np.full(shape=(T, mc_sim), fill_value=0.0)

initial_value_of_portfolio = 10000

for m in range(0, mc_sim):
    Z = np.random.normal(size=(T, len(weight_of_stocks)))
    L = np.linalg.cholesky(cov_Mat)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_no_sim[:,m] = np.cumprod(np.dot(weight_of_stocks, dailyReturns)+1)*initial_value_of_portfolio


Return = pd.Series(portfolio_no_sim[-1,:])

VaR = initial_value_of_portfolio - mcVaR(Return, alpha=5)

if list_of_stocks == ['KO', 'GOOG', 'GM']:
    vr_x = 30
    vr_y = 9200
elif list_of_stocks == ['AAPL', 'GOOG', 'MSFT']:
    vr_x = 30
    vr_y = initial_value_of_portfolio - VaR + 650



plt.hlines(initial_value_of_portfolio - VaR, xmin = 0, xmax = 100, color = "black", zorder = 1000000)
plt.hlines(initial_value_of_portfolio, xmin = 0, xmax = 100, color = "black", zorder = 1000000)
plt.text(vr_x,vr_y, r'VaR = %.2f[\$]'%(VaR), fontsize = 13, zorder = 1000000)

plt.plot(portfolio_no_sim)
plt.ylabel('Value of the portfolio [USD]')
plt.xlabel('Days')
plt.title('Monte Carlo simulation of stock portfolio')
plt.show()

print('VaR ${}'.format(round(VaR,2)))
