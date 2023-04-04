import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.stats import zscore

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
data.drop('timestamp', axis=1, inplace=True)

def get_data_matrices(train_data, test_data, fold):
    # Removing redundancies in time complexity using this function, similarity matrix is calculated only once
    
    data_matrix = np.zeros((data.user_id.max(), data.item_id.max())) # Matrix with rows as users and columns as items (0-indexed)
    for line in train_data.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]

    # Store the similarity scores array in a file in the sims folder if it doesn't exist
    if not os.path.exists(f'sims/user_similarity_{fold}.npy'):
        user_similarity = np.zeros((data.user_id.max(), data.user_id.max()))
        for i in tqdm(range(data.user_id.max())):
            for j in range(data.user_id.max()):
                user_similarity[i,j] = cosine_similarity(data_matrix[i,:].reshape(1,-1), data_matrix[j,:].reshape(1,-1))
        np.save(f'sims/user_similarity_{fold}.npy', user_similarity)
    else:
        user_similarity = np.load(f'sims/user_similarity_{fold}.npy')

    return data_matrix, user_similarity

def get_mae(train_data, test_data, k, data_matrix, user_similarity, weighting='mean'):
    """
    Returns the mean absolute error of the predictions for the test data

    Note: The Assumption that in case if none of the neighbors have rated the item, the mean of the non-zero ratings of the user itself for that item is used

    weihgting parameter can be one of the following:
    'mean' - Mean of the ratings of the neighbors for the item
    'similarity' - Weighted mean of the ratings of the neighbors for the item, where the weights are the similarity scores of the neighbors
    'significance' - Weighted mean of the ratings of the neighbors for the item, where the weights are multiplied by the number of ratings of the neighbors

    """

    users_tbc = []
    # Iterate over the test data and store the user ids of the users that need to be predicted
    for line in test_data.itertuples():
        if line[1] not in users_tbc:
            users_tbc.append(line[1])
    
    neighbor_ratings = {} # Should be user_id : [k x 1682 shaped matrix]
    neighbor_similarity = {} # Should be user_id : [k length vector]
    neighbor_indices = {} # Should be user_id : [k length vector]
    for user in users_tbc:
        # Get the indices of the k=10 neighbors except the user itself with the highest similarity scores
        neighbors = np.argsort(user_similarity[user-1,:])[-k-1:-1]
        neighbor_indices[user] = neighbors

        # Get the ratings of the neighbors for each item
        neighbor_ratings[user] = data_matrix[neighbors,:]
        neighbor_similarity[user] = user_similarity[user-1,neighbors]

    predicted_data = test_data.copy()
    predicted_data['predicted_rating'] = 0

    for line in tqdm(predicted_data.itertuples(), total=len(predicted_data)):
        cur_user = line[1]
        cur_item = line[2]

        # Find the mean of the ratings of the neighbors for the current item for non-zero ratings
        if weighting == 'mean':
            mean_neighbor_rating = 0
            useful_neighbors = 0
            for i in range(k):
                if neighbor_ratings[cur_user][i,cur_item-1] != 0:
                    mean_neighbor_rating += neighbor_ratings[cur_user][i,cur_item-1]
                    useful_neighbors += 1
            if useful_neighbors != 0:
                mean_neighbor_rating /= useful_neighbors
            else:
                # If none of the neighbors have rated the item, use the mean of the non-zero ratings of the user itself for that item
                mean_neighbor_rating = np.mean(data_matrix[cur_user-1, :][data_matrix[cur_user-1, :] != 0])

        elif weighting == 'similarity':
            mean_neighbor_rating = 0
            similarity_denominator = 0
            useful_neighbors = False
            for i in range(k):
                if neighbor_ratings[cur_user][i,cur_item-1] != 0:
                    mean_neighbor_rating += neighbor_ratings[cur_user][i,cur_item-1] * neighbor_similarity[cur_user][i]
                    similarity_denominator += neighbor_similarity[cur_user][i]
                    useful_neighbors = True
            if useful_neighbors:
                mean_neighbor_rating /= similarity_denominator
            else:
                mean_neighbor_rating = np.mean(data_matrix[cur_user-1, :][data_matrix[cur_user-1, :] != 0])

        elif weighting == 'significance':
            mean_neighbor_rating = 0
            denominator = 0
            useful_neighbors = False
            for i in range(k):
                if neighbor_ratings[cur_user][i,cur_item-1] != 0:
                    # Find the number of items that the neighbor has rated that the current user has also rated for item_matches
                    item_matches = np.count_nonzero(data_matrix[cur_user-1, :] * neighbor_ratings[cur_user][i,:])
                    significance_denominator = 40
                    significance_weight = item_matches / significance_denominator
                    if significance_weight > 1:
                        significance_weight = 1
                    mean_neighbor_rating += neighbor_ratings[cur_user][i,cur_item-1] * neighbor_similarity[cur_user][i] * significance_weight
                    denominator += neighbor_similarity[cur_user][i] * significance_weight
                    useful_neighbors = True
            if useful_neighbors:
                mean_neighbor_rating /= denominator
            else:
                mean_neighbor_rating = np.mean(data_matrix[cur_user-1, :][data_matrix[cur_user-1, :] != 0])
                
        predicted_data.at[line[0], 'predicted_rating'] = round(mean_neighbor_rating)

    mae = 0
    for line in predicted_data.itertuples():
        mae += abs(line[3] - line[4])
    mae /= len(predicted_data)

    return mae

results = {}

for i in range(1, 6):
    print(f"Fold {i}: ")
    results[i] = {}

    train_data = pd.read_csv(f'ml-100k/u{i}.base', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_data = pd.read_csv(f'ml-100k/u{i}.test', sep="\t", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    train_data.drop('timestamp', axis=1, inplace=True)
    test_data.drop('timestamp', axis=1, inplace=True)
    
    print(f"Getting data matrices and user similarity matrix for fold {i}... ")
    tic = time.time()
    data_matrix, user_similarity = get_data_matrices(train_data, test_data, fold=i)
    toc = time.time()
    print(f"Time taken to get data matrices for fold {i}: ", toc-tic)

    for k in range(10, 60, 10):
        print("Calculating MAE for k = ", k)
        mae = get_mae(train_data, test_data, k, data_matrix, user_similarity, weighting='similarity')
        print(f"MAE for fold {i} and k neighbors = {k}: ", mae)
        print()
        results[i][k] = mae
        
    print('-'*50)
    print()

print("User-User Collaborative Filtering with Weighted Similarities:")
print("\nResults: ", end='\t')
for k in range(10, 60, 10):
    print(f'k = {k}', end='\t\t')
print()
for i in range(1, 6):
    print(f"Fold {i}", end='\t\t')
    for k in range(10, 60, 10):
        # Print result rounded to 2 decimal places
        print(round(results[i][k], 4), end='\t\t')
    print()
print("Average", end='\t\t')
for k in range(10, 60, 10):
    # Print result rounded to 2 decimal places
    print(round(np.mean([results[i][k] for i in range(1, 6)]), 4), end='\t\t')