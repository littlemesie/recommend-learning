import numpy as np
from keras.utils import to_categorical

data1 = [1, 2, 3, 4, 5, 1, 4, 2]
data1 = np.array(data1)

# 有普通np数组转换为one-hot
one_hots1 = to_categorical(data1)
print(one_hots1)

data2 = [1, 2, 3, 4, 5, 1, 4, 2]
data2 = np.array(data2)

# 有普通np数组转换为one-hot
one_hots2 = to_categorical(data2)
print(one_hots2)

a = np.concatenate((one_hots1, one_hots2), axis=1)
# print(a)