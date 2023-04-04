import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import wandb

np.random.seed(19)

parser = argparse.ArgumentParser()
parser.add_argument('--outer', type=int, default=200, help='Number of outer iterations')
parser.add_argument('--inner', type=int, default=6, help='Number of inner iterations')
args = parser.parse_args()

wandb.init(project='cf_a2', config=args)

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
data.drop('timestamp', axis=1, inplace=True)
data.describe()

user_count = data['user_id'].nunique()
item_count = data['item_id'].nunique()
print('user_count: ', user_count)
print('item_count: ', item_count)

def get_mae(train_data, test_data, factor_count, inner_iter, outer_iter):
    np.random.seed(0)
    # Creating X ratings matrix X0 for initialisation of ALS algorithm with random values
    factor_count = factor_count
    u0 = np.random.rand(user_count, factor_count)
    v0 = np.random.rand(item_count, factor_count)
    X0 = np.random.randint(low=1, high=6, size=(user_count, item_count))
    # We will now create the R mask matrix which stores 1 for ratings present and 0 for ratings not present
    # Also, we will create the Y ratings matrix which stores the actual ratings
    R = np.zeros((user_count, item_count))
    Y = np.zeros((user_count, item_count))
    for row in train_data.itertuples():
        R[row[1]-1, row[2]-1] = 1
        Y[row[1]-1, row[2]-1] = row[3]

    print("Working for factor_count = ", factor_count)
    # Now we want to minimize ||Y - R.(UV^T)||^2_F where X = UV^T

    max_iter = outer_iter
    inner_iter = inner_iter
    mae_values = {}

    cur_X = X0
    cur_u = u0
    cur_v = v0
    for k in tqdm(range(max_iter)):
        B_k = cur_X + Y - R*np.dot(cur_u, cur_v.T)
        # Now we want to minimize ||B_k - U_k.V_k^T||^2_F using ALS
        u_j = cur_u
        v_j = cur_v
        for j in range(inner_iter):
            # Using numpy's solve function (which uses LU decomposition) to solve the linear system
            v_j = np.linalg.solve(np.dot(u_j.T, u_j), np.dot(u_j.T, B_k))
            u_j = np.linalg.solve(np.dot(v_j, v_j.T), np.dot(v_j, B_k.T)).T
            # Using numpy's lstsq function to solve the linear system
            # v_j = np.linalg.lstsq(np.dot(u_j.T, u_j), np.dot(u_j.T, B_k))[0]
            # u_j = np.linalg.lstsq(np.dot(v_j, v_j.T), np.dot(v_j, B_k.T))[0].T
        cur_u = u_j
        cur_v = v_j.T
        cur_X = np.dot(cur_u, cur_v.T)

    # Now we will calculate the NMAE for the test data
    mae = 0
    count = 0
    min_rate = 5
    max_rate = 1
    for row in test_data.itertuples():
        mae += abs(row[3] - cur_X[row[1]-1, row[2]-1])
        count += 1
        if row[3] < min_rate:
            min_rate = row[3]
        if row[3] > max_rate:
            max_rate = row[3]
    mae /= count
    print('For factor_count = ', factor_count, ', Inner Iterations = ', inner_iter, ', Outer Iterations = ', outer_iter,', MAE = ', mae)
    nmae = mae/(max_rate - min_rate)
    return nmae


# Running loop for filling the table

results = {}

for fold in range(1,6):
    print('Working for fold ', fold)
    train_data = pd.read_csv(f'ml-100k/u{fold}.base', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_data = pd.read_csv(f'ml-100k/u{fold}.test', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    train_data.drop('timestamp', axis=1, inplace=True)
    test_data.drop('timestamp', axis=1, inplace=True)
    for factor_count in range(1,11):
        nmae = get_mae(train_data, test_data, factor_count, args.inner, args.outer)
        if fold not in results:
            results[fold] = {}
        results[fold][factor_count] = nmae

# Pretty printing the results
print("\nResults", end='\t|\t')
for factor in range(1,11):
    print(f'Param {factor}', end='\t|\t')
print()
for i in range(1, 6):
    print(f"Fold {i}", end='\t|\t')
    for k in range(1,11):
        # Print result rounded to 2 decimal places
        print(round(results[i][k], 5), end='\t|\t')
    print()
print("Average", end='\t|\t')
for k in range(1,11):
    # Print result rounded to 2 decimal places
    print(round(np.mean([results[i][k] for i in range(1, 6)]), 5), end='\t|\t')


