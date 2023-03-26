import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

stock_data = pd.read_csv('stock.csv', index_col=0)
trading_days = 252

def plot(stock_data, name):

    plt.figure(1)
    plt.plot(stock_data['Apple' + name], label='Apple')
    plt.plot(stock_data['Coke' + name], label='Coke')
    plt.plot(stock_data['SP500' + name], label='SP500')
    plt.legend()

    if name == '':
        plt.figure(2)
        plt.plot(stock_data['Apple'], label='Apple')
        plt.plot(stock_data['Coke'], label='Coke')
        plt.legend()

        plt.figure(3)
        plt.plot(stock_data['Coke'], label='Coke')
        plt.legend()
    plt.show()

## part a)
# plot(stock_data, '')


stocks = ['Apple', 'SP500', 'Coke']
for stock in stocks:
    stock_data[stock + 'shifted'] = stock_data[stock].shift(-1)


stock_data.drop(stock_data.index[-1], inplace=True)
# calculate Day-to day return
for stock in stocks:
    stock_data[stock + 'Day-to-Day_return'] = stock_data.apply(lambda row: (row[stock] - row[stock + 'shifted']) / row[stock], axis=1)
# plot day to day return
# plot(stock_data, 'Day-to-Day_return')
# caluclate average of day to day
for stock in stocks:
    locals()[stock+'average'] = stock_data[stock + 'Day-to-Day_return'].mean()*(trading_days-1)/trading_days


# calculate the covariance matrix

for stock in stocks:
    stock_data[stock + 'standarized'] = stock_data[stock + 'Day-to-Day_return'] - locals()[stock+'average']

tmp = list(zip(stock_data['Applestandarized'], stock_data['SP500standarized'], stock_data['Cokestandarized']))
tmp = pd.DataFrame(data=tmp)
tmp = tmp.to_numpy()

covariant_matrix = np.matmul(np.transpose(tmp), tmp)/trading_days
inv_covariant_matrix = np.invert(covariant_matrix)

# Apply Markowitz Theory without risk free assets.
# Lagrange Multiplier: 1/2 sum_i sum_j C_ij w_i w_j + lambda_1 [rho - sum_i w_i <r_i>] + lambda_2[1 - sum_i w_i] + lambda_2[0.3 - w_coca cola]
#
#


def function(cov_matrix, weights, average_values, profit, lambdas):
    a = np.einsum('i, ij, j', weights, cov_matrix, weights)
    b = lambdas[0](profit - np.matmul(weights, np.transpose(average_values)))
    c = lambdas[1](1 - weights.sum())
    d = lambdas[2](0.3 - weights[2])
    return a + b + c + d
