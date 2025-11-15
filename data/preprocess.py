"""Preprocessing utilities for MovieLens-1M"""

import os
import numpy as np
import pandas as pd

def load_movielens(path="data/movielens-1m/ratings.dat", min_interactions=5):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ratings file not found at {path}. Run data/download_datasets.sh")

    df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "rating", "ts"])
    user_counts = df['user'].value_counts()
    valid_users = set(user_counts[user_counts >= min_interactions].index)
    df = df[df['user'].isin(valid_users)]

    item_counts = df['item'].value_counts()
    valid_items = set(item_counts[item_counts >= 5].index)
    df = df[df['item'].isin(valid_items)]

    unique_users = sorted(df['user'].unique())
    unique_items = sorted(df['item'].unique())
    user_map = {old: new for new, old in enumerate(unique_users)}
    item_map = {old: new for new, old in enumerate(unique_items)}

    df = df.copy()
    df['user'] = df['user'].map(user_map)
    df['item'] = df['item'].map(item_map)
    df['rating'] = df['rating'].astype(float) / 5.0

    users = df['user'].to_numpy(dtype=np.int32)
    items = df['item'].to_numpy(dtype=np.int32)
    ratings = df['rating'].to_numpy(dtype=np.float32)

    data = {
        'users': users,
        'items': items,
        'ratings': ratings,
        'n_users': len(unique_users),
        'n_items': len(unique_items),
        'df': df
    }
    return data

def create_train_test_split(data, test_ratio=0.2, seed=0):
    np.random.seed(seed)
    df = data['df']
    train_rows = []
    test_rows = []

    for user, group in df.groupby('user'):
        n = len(group)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_test = max(1, int(np.floor(test_ratio * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_rows.append(group.iloc[test_idx])
        train_rows.append(group.iloc[train_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    train = {
        'users': train_df['user'].to_numpy(dtype=np.int32),
        'items': train_df['item'].to_numpy(dtype=np.int32),
        'ratings': train_df['rating'].to_numpy(dtype=np.float32)
    }
    test = {
        'users': test_df['user'].to_numpy(dtype=np.int32),
        'items': test_df['item'].to_numpy(dtype=np.int32),
        'ratings': test_df['rating'].to_numpy(dtype=np.float32)
    }
    return train, test

if __name__ == "__main__":
    try:
        data = load_movielens()
        print(f"Users: {data['n_users']}, Items: {data['n_items']}, Ratings: {len(data['ratings'])}")
    except FileNotFoundError as e:
        print(str(e))
        print("You can run data/download_datasets.sh to fetch MovieLens-1M")
