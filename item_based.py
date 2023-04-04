import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
data.drop('timestamp', axis=1, inplace=True)

def get_data_matrices(train_data, test_data, fold):

    data_matrix = np.zeros((data.user_id.max(), data.item_id.max())) # Matrix with rows as users and columns as items (0-indexed)
    for line in train_data.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]

    if not os.path.exists(f'sims/item_similarity_{fold}.npy'):
        item_similarity = np.zeros((data.item_id.max(), data.item_id.max()))
        for i in tqdm(range(data.item_id.max())):
            for j in range(data.item_id.max()):
                item_similarity[i,j] = cosine_similarity(data_matrix[:,i].reshape(1,-1), data_matrix[:,j].reshape(1,-1))
        np.save(f'sims/item_similarity_{fold}.npy', item_similarity)
    else:
        item_similarity = np.load(f'sims/item_similarity_{fold}.npy')

    return data_matrix, item_similarity

def get_mae(train_data, test_data, k, data_matrix, item_similarity, weighting='mean'):

    items_tbc = []
    # Iterate over the test data and store the item ids of the items that need to be predicted
    for line in test_data.itertuples():
        if line[2] not in items_tbc:
            items_tbc.append(line[2])
    
    neighbor_ratings = {} # Should be item_id : [943 x k shaped matrix]
    neighbor_similarity = {} # Should be item_id : [k length vector]
    neighbor_indices = {} # Should be item_id : [k length vector]
    for item in items_tbc:
        # Get the indices of the k=10 neighbor items except the item itself with the highest similarity scores
        neighbors = np.argsort(item_similarity[item-1,:])[-k-1:-1]
        neighbor_indices[item] = neighbors
        # Get the ratings of the neighbors for each item
        neighbor_ratings[item] = data_matrix[:,neighbors]
        neighbor_similarity[item] = item_similarity[item-1,neighbors]

    predicted_data = test_data.copy()
    predicted_data['predicted_rating'] = 0

    if weighting == 'variance':

        item_variance = {}
        for item_id in range(1, data.item_id.max()+1):
            var_item = np.var(data_matrix[:,item_id-1][data_matrix[:,item_id-1] != 0])
            if np.isnan(var_item):
                var_item = 0
            item_variance[item_id] = var_item

        varmin = min(item_variance.values())
        varmax = max(item_variance.values())

        item_vi = {}
        for item_id in range(1, data.item_id.max()+1):
            item_vi[item_id] = (item_variance[item_id] - varmin) / varmax
            

    for line in tqdm(predicted_data.itertuples(), total=len(predicted_data)):
        cur_user = line[1]
        cur_item = line[2]
        # Find the mean of the ratings of the neighbors for the current item for non-zero ratings

        if weighting == 'mean':
            mean_neighbor_rating = 0
            useful_neighbors = 0
            for i in range(k):
                if neighbor_ratings[cur_item][cur_user-1, i] != 0:
                    mean_neighbor_rating += neighbor_ratings[cur_item][cur_user-1,i]
                    useful_neighbors += 1
            if useful_neighbors != 0:
                mean_neighbor_rating /= useful_neighbors
            else:
                # If none of the neighbors have rated the item, use the mean of the non-zero ratings of the item itself for that user
                mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                # Check if it is nan
                if np.isnan(mean_neighbor_rating):
                    mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])

        elif weighting == 'similarity':
            mean_neighbor_rating = 0
            similarity_denominator = 0
            useful_neighbors = False
            for i in range(k):
                if neighbor_ratings[cur_item][cur_user-1, i] != 0:
                    mean_neighbor_rating += neighbor_ratings[cur_item][cur_user-1,i] * neighbor_similarity[cur_item][i]
                    similarity_denominator += neighbor_similarity[cur_item][i]
                    useful_neighbors = True
            if useful_neighbors:
                if similarity_denominator == 0: # Case where all the neighbors have similarity 0
                    mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                    if np.isnan(mean_neighbor_rating):
                        mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])
                else:
                    mean_neighbor_rating /= similarity_denominator
            else:
                # If none of the neighbors have rated the item, use the mean of the non-zero ratings of the item itself for that user
                mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                # Check if it is nan
                if np.isnan(mean_neighbor_rating):
                    mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])
            

        elif weighting == 'significance':
            mean_neighbor_rating = 0
            denominator = 0
            useful_neighbors = False
            for i in range(k):
                if neighbor_ratings[cur_item][cur_user-1, i] != 0:
                    # Matches - Find the number of users who have rated the item and the neighbor item
                    matches = np.count_nonzero(data_matrix[:,cur_item-1] * neighbor_ratings[cur_item][:,i])
                    significance_denominator = 50
                    significance_weight = matches / significance_denominator
                    if significance_weight > 1:
                        significance_weight = 1
                    mean_neighbor_rating += neighbor_ratings[cur_item][cur_user-1,i] * neighbor_similarity[cur_item][i] * significance_weight
                    denominator += neighbor_similarity[cur_item][i] * significance_weight
                    useful_neighbors = True
            if useful_neighbors:
                if denominator == 0: # Case where all the neighbors have similarity 0
                    mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                    if np.isnan(mean_neighbor_rating):
                        mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])
                else:
                    mean_neighbor_rating /= denominator
            else:
                # If none of the neighbors have rated the item, use the mean of the non-zero ratings of the item itself for that user
                mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                # Check if it is nan
                if np.isnan(mean_neighbor_rating):
                    mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])

        elif weighting == 'variance':
            mean_neighbor_rating = 0
            denominator = 0
            useful_neighbors = False
            for i in range(k):
                if neighbor_ratings[cur_item][cur_user-1, i] != 0:
                    mean_neighbor_rating += neighbor_ratings[cur_item][cur_user-1,i] * neighbor_similarity[cur_item][i] * item_vi[neighbor_indices[cur_item][i]+1]
                    denominator += neighbor_similarity[cur_item][i] * item_vi[neighbor_indices[cur_item][i]+1]
                    useful_neighbors = True
            if useful_neighbors:
                if denominator == 0:
                    mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                    if np.isnan(mean_neighbor_rating):
                        mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])
                else:
                    mean_neighbor_rating /= denominator
            else:
                # If none of the neighbors have rated the item, use the mean of the non-zero ratings of the item itself for that user
                mean_neighbor_rating = np.mean(data_matrix[:,cur_item-1][data_matrix[:,cur_item-1] != 0])
                if np.isnan(mean_neighbor_rating):
                    mean_neighbor_rating = np.mean(data_matrix[cur_user-1,:][data_matrix[cur_user-1,:] != 0])

        
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
    data_matrix, item_similarity = get_data_matrices(train_data, test_data, fold=i)
    toc = time.time()
    print(f"Time taken to get data matrices for fold {i}: ", toc-tic)

    for k in range(10, 60, 10):
        print("Calculating MAE for k = ", k)
        mae = get_mae(train_data, test_data, k, data_matrix, item_similarity, weighting='variance')
        print(f"MAE for fold {i} and k neighbors = {k}: ", mae)
        print()
        results[i][k] = mae
        
    print('-'*50)
    print()

print("Item-Item Collaborative Filtering with Variance Weighting")
print("\nResults", end='\t\t')
for k in range(10, 60, 10):
    print(f'k = {k}', end='\t\t')
print()
for i in range(1, 6):
    print(f"Fold {i}", end='\t\t')
    for k in range(10, 60, 10):
        # Print result rounded to 2 decimal places
        print(round(results[i][k], 5), end='\t\t')
    print()
print("Average", end='\t\t')
for k in range(10, 60, 10):
    # Print result rounded to 2 decimal places
    print(round(np.mean([results[i][k] for i in range(1, 6)]), 5), end='\t\t')
    