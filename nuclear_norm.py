# In this, we will implement the nuclear norm minimization algorithm for Collaborative Filtering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
data.drop('timestamp', axis=1, inplace=True)

user_count = data['user_id'].nunique()
item_count = data['item_id'].nunique()
print('user_count: ', user_count)
print('item_count: ', item_count)

# Load the data - MovieLens 100K
train_data = pd.read_csv(f'ml-100k/u1.base', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
test_data = pd.read_csv(f'ml-100k/u1.test', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
train_data.drop('timestamp', axis=1, inplace=True)
test_data.drop('timestamp', axis=1, inplace=True)

# Create a matrix of users and items - Y which is what the users have actually rated
# Create a matrix of users and items - R which is a binary matrix where 1 indicates that the user has rated the item
Y = np.zeros((user_count, item_count))
R = np.zeros((user_count, item_count))
for row in train_data.itertuples():
    Y[row[1]-1, row[2]-1] = row[3]
    R[row[1]-1, row[2]-1] = 1

# Now initialzing X, the matrix to be predicted (the oracle)
X0 = np.random.randint(1, 6, size=(user_count, item_count))
epochs = 1000
num_features = 10
lambda_ = 0.1

# Now we will implement the nuclear norm minimization algorithm
X_cur = X0
for epoch in tqdm(range(epochs)):
    B = X_cur + Y - R*X_cur
    U, s, V = np.linalg.svd(B)
    svs = s.shape[0]
    s_hat = np.maximum(s - lambda_/2, 0)
    S = np.zeros((user_count, item_count))
    S[:svs, :svs] = np.diag(s_hat)
    X_cur = np.dot(U, np.dot(S, V))

# Now we will calculate the NMAE
mae = 0
count = 0
min_rate = 5
max_rate = 1
for row in test_data.itertuples():
    mae += abs(row[3] - X_cur[row[1]-1, row[2]-1])
    count += 1
    if row[3] < min_rate:
        min_rate = row[3]
    if row[3] > max_rate:
        max_rate = row[3]
mae /= count
nmae = mae/(max_rate - min_rate)
print('MAE: ', mae)
print('NMAE: ', nmae)

