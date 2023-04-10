from numpy.core.multiarray import dot
from numpy.ma.extras import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# to ignore warings when handling dataframes
# import warnings
# warnings.filterwarnings("ignore")


# debugging for better readability on the terminal
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))


stock_data = pd.read_csv('stock.csv', index_col=0)
trading_days = 252

def plot(stock_data, name, stocjs):

    plt.figure()
    for stock in stocks:

        plt.plot(stock_data[stock + name], label=stock, linewidth=0.7)
    plt.legend()

    if name == '':
        plt.figure()
        plt.plot(stock_data['Apple'], label='Apple', linewidth=0.7)
        plt.plot(stock_data['Coke'], label='Coke', linewidth=0.7)
        plt.legend()

        plt.figure()
        plt.plot(stock_data['Coke'], label='Coke', linewidth=0.7)
        plt.legend()
    # plt.show()

## part a)
stocks = ['Apple', 'SP500', 'Coke']
plot(stock_data, '', stocks)
for stock in stocks:
    stock_data[stock + 'shifted'] = stock_data[stock].shift(-1)

stock_data.drop(stock_data.index[-1], inplace=True)

# calculate Day-to day return
for stock in stocks:
    stock_data[stock + 'Day-to-Day_return'] = stock_data.apply(lambda row: (row[stock + 'shifted'] - row[stock]) / row[stock], axis=1)


# plot day to day return: uncomment following line
plot(stock_data, 'Day-to-Day_return', stocks)


# caluclate average of day to day
average_return = np.zeros(3)
cnt=0
for stock in stocks:
    locals()[stock+'average'] = stock_data[stock + 'Day-to-Day_return'].mean()
    average_return[cnt] = stock_data[stock + 'Day-to-Day_return'].mean()
    cnt += 1

# calculate the covariance matrix
for stock in stocks:
    stock_data[stock + 'standarized'] = stock_data[stock + 'Day-to-Day_return'] - locals()[stock+'average']

tmp = list(zip(stock_data['Applestandarized'], stock_data['SP500standarized'], stock_data['Cokestandarized']))
tmp = pd.DataFrame(data=tmp)
tmp = tmp.to_numpy()

covariant_matrix = np.matmul(np.transpose(tmp), tmp)/trading_days
inv_covariant_matrix = inv(covariant_matrix)


def plt_volatility_return(covariant_matrix, average_return, stocks):
    for k in range(3):
        plt.scatter(np.sqrt(covariant_matrix[k][k]), average_return[k], label=stocks[k])
    plt.legend()
    plt.xlabel('volatility')
    plt.ylabel('return')

# plot volatily of individual stocks: uncomment following line
plt_volatility_return(covariant_matrix, average_return, stocks)


# Apply Markowitz Theory without risk free assets.
returns = np.array([0.05, 0.1, 0.15])/trading_days
e = np.ones(3)
e3 = np.zeros(3)
e3[2] = 1

def portfolio(rtrn, inv_covariant_matrix, v, inv_A):
    w = np.array([rtrn, 1, 0.3])
    lambdas = dot(inv_A, w)
    weights = lambdas[0]*np.matmul(inv_covariant_matrix, v[0]) + lambdas[1]*np.matmul(inv_covariant_matrix, v[1]) + lambdas[2]*np.matmul(inv_covariant_matrix, v[2])
    return weights


def volatility(returns, inv_covariant_matrix, v, covariant_matrix):
    volatilities = np.zeros(len(returns))
    A_ij = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            A_ij[i][j] = dot(v[i], np.matmul(inv_covariant_matrix, v[j]))
    inv_A = inv(A_ij)
    for k in range(len(returns)):
        weights = portfolio(returns[k], inv_covariant_matrix, v, inv_A)
        volatilities[k] = np.sqrt(np.matmul(weights, np.matmul(covariant_matrix, weights)))
    return volatilities


plt.figure()
returns = np.linspace(0, 0.275, 101)/trading_days

v = np.array([average_return, e, e3])
volatilities = volatility(returns, inv_covariant_matrix, v, covariant_matrix)

plt.plot(volatilities, returns)
plt_volatility_return(covariant_matrix, average_return, stocks)
plt.show()
