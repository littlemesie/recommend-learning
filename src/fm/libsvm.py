from svmutil import *

y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
prob = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')