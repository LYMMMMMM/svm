from numpy import *

x = array([[1, 0, 0],
           [0, 1, 1]])

y = arange(12).reshape(2, 6)
def f(x):
    return -x
print(abs(-2))
a = globals().get('f')


x[array([0])] = 0
print(x)

