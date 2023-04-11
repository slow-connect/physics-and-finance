from numpy.core.function_base import linspace
from numpy import zeros
from scipy.integrate import quad
from math import pi, exp, log, sqrt, inf
from matplotlib.pyplot import plot, xlabel, ylabel, show


def psi(s, si, rho, t):
    a = 1/sqrt(2*pi*si*si*t)
    b = (-1) * (log(s) - rho*t)*(log(s) - rho*t) / (2*si*si*t)
    return a * exp(b)/s

si = 0.3
rho = 0.1
t = 2
int = quad(psi, 0.4, 0.6, args=(si, rho, t))
print(int)

x = linspace(0.0001, 5, 1001)
y = zeros(len(x))
for k in range(len(x)):
    y[k] = psi(x[k], si, rho, t)
plot(x, y)
xlabel(r'value $s$ of stock')
ylabel(r'probability amplitude $\Psi$')
show()
