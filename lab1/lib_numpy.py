import numpy as np


arr_1 = np.zeros(10)    # 1

arr_2 = np.zeros(10)    # 2
arr_2[4] = 1

arr_3 = np.random.random(20)    # 3
indices_arr_3 = np.nonzero(arr_3)

arr_4 = np.random.random((3, 3, 3))    # 4

arr_5 = np.random.random(10)    # 5
mean_arr_5 = np.mean(arr_5)

arr_6_1 = np.random.random((5, 3))    # 6
arr_6_2 = np.random.random((3, 2))
prod_6 = np.dot(arr_6_1, arr_6_2)

arr_7_1 = np.random.random((4, 4))    # 7
arr_7_2 = np.random.random((4, 4))
prod_7 = np.dot(arr_7_1, arr_7_2)
diag_7 = np.diag(prod_7)

arr_8 = np.random.random(20)    # 8
max_idx_8 = np.argmax(arr_8)
arr_8[max_idx_8] = 0

arr_9 = np.random.random(20)    # 9
unique_9 = np.unique(arr_9)

mtrx_10 = np.random.random((3, 3))    # 10
mtrx_10_mean = mtrx_10.mean()
new_mtrx_10 = mtrx_10 - mtrx_10_mean

mtrx_11 = np.random.random((3, 3))    # 11
mtrx_11[[0, 1]] = mtrx_11[[1, 0]]

arr_12 = np.random.random(20)    # 12
n_12 = 5
top_n_12 = np.sort(arr_12)[-n_12:]

mtrx_13 = np.random.randint(1, 11, (5, 5))    # 13
row_sums_13 = mtrx_13.sum(axis=1)

arr_14 = np.random.uniform(-1, 1, 10)    # 14
arr_14 = np.where(arr_14 > 0, 1, np.where(arr_14 < 0, -1, 0))

arr_15 = np.random.randint(1, 101, 12)    # 15
parts_15 = np.split(arr_15, 3)
sums = [p.sum() for p in parts_15]
