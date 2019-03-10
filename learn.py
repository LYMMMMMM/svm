from numpy import *

x = array([[1, 2, 3, 4],
           [5, 4, 3, 2]])
y = array([[1],[2],[3],[4]])
xt = x[[0], :]
print(sum(y))
print( x[1, 2] )
