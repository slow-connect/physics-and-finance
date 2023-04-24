import matplotlib.pyplot as plt
from numpy import matmul, array, linspace, polyval, sqrt, identity, diag
from scipy.linalg import inv
import pandas as pd
from scipy.integrate import quad



def prCol(skk): print("\033[93m {}\033[00m" .format(skk))

def pinv(mat):
    return inv(matmul(mat.T, mat))

def chi(y, x, param, si):
    val = ((y - polyval(param, x))/si)**2
    # print(val)
    return sum(val)

def beta(z, w):
    def f(t, z, w):
        return t**(z-1)* (1 - t)**(w-1)
    return quad(f, 0, 1, args=(z, w))[0]
def beta_x(a, b, x):
    def f(t, a, b):
        return t**(a-1)* (1 - t)**(b-1)
    return quad(f, 0, x, args=(a, b))[0]


def inc_beta(x, a, b):
    return beta_x(a, b, x) / beta(a, b)



data = pd.read_csv('polynomial_data.csv')
data_copy = pd.read_csv('polynomial_data.csv', index_col=0)

# part (a)
def plotdata(data):
    plt.plot(data, '.')
    plt.xlabel(r'x')
    plt.ylabel(r'y')


# uncomment to plot data
plotdata(data_copy)


# part (b)
DEG_POLY = 3
prCol(DEG_POLY)

col_names = ['x to pow 3', 'x to pow 2', 'x to pow 1', 'x to pow 0']
N = len(data)
for k in range(DEG_POLY+1):
    data['x to pow ' + str(k)] = data.apply(lambda row: row['x']**k, axis=1)
x = array(data[['x to pow 1']].values)
y = array(data[['y']].values)
mat = array(data[col_names].values)
yy = matmul(mat.T, y)
param = matmul(pinv(mat), yy)
poly3 = param
prCol(param)

def plot_poly_points(x, param, data):
    xx = linspace(min(x), max(x), 101)
    plt.plot(xx, polyval(param, xx))
    plotdata(data)


# uncomment for points + polynomial
plt.figure()
plot_poly_points(x, param, data_copy)


# part (c)

yy = polyval(param, x)
err = (yy - y)**2
err_bar = sqrt(1/len(err)*sum(err))
print(err_bar)

# part (d)

err_matrix = 1/err_bar * identity(N)
cov = pinv(matmul(err_matrix, mat))
err_param = sqrt(diag(cov))

# print(cov)
prCol(err_param)

p3 = chi(y, x, param, err_bar)
print(p3)

# part (e)

DEG_POLY = 2
prCol(DEG_POLY)
col_names = ['x to pow 2', 'x to pow 1', 'x to pow 0']
N = len(data)
for k in range(DEG_POLY+1):
    data['x to pow ' + str(k)] = data.apply(lambda row: row['x']**k, axis=1)
x = array(data[['x to pow 1']].values)
y = array(data[['y']].values)
mat = array(data[col_names].values)
yy = matmul(mat.T, y)
param = matmul(pinv(mat), yy)
prCol(param)

plt.figure()
plot_poly_points(x, param, data_copy)
yy = polyval(param, x)
err = (yy - y)**2
err_bar = sqrt(1/len(err)*sum(err))
print(err_bar)


err_matrix = 1/err_bar * identity(N)
cov = pinv(matmul(err_matrix, mat))
err_param = sqrt(diag(cov))

# print(cov)
prCol(err_param)
p2 = chi(y, x, param, err_bar)
print(p2)
f = (p3 - p2)/(p3*(N - 3))
print()
print(f)
m = N - 3
n = 1

x_ = n * f / (m + n * f)

prob = inc_beta(x_, n/2, m/2)
print(prob)

plt.figure()
plot_poly_points(x, param, data_copy)
xx = linspace(min(x), max(x), 101)
plt.plot(xx, polyval(poly3, xx))
plt.show()
