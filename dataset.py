import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

def load_ratings(path: str, min_rating: int = 4):
    user_set, item_set = set(), set()
    raw_positives = []

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            uid, iid, rating = int(parts[0]), int(parts[1]), float(parts[2])
            if rating >= min_rating:
                raw_positives.append((uid, iid))
                user_set.add(uid)
                item_set.add(iid)

    user2idx = {u: i for i, u in enumerate(sorted(user_set))}
    item2idx = {it: i for i, it in enumerate(sorted(item_set))}

    positives = [(user2idx[u], item2idx[i]) for u, i in raw_positives]
    return positives, user2idx, item2idx, len(user2idx), len(item2idx)



def sample_negatives(positives: list, num_items: int, neg_ratio: int = 4, seed: int = 42):
    rng = random.Random(seed)
    user_positives = defaultdict(set)
    for u, i in positives:
        user_positives[u].add(i)

    all_items = list(range(num_items))
    samples = []

    for u, i in positives:
        samples.append((u, i, 1))
        seen = user_positives[u]
        count = 0
        while count < neg_ratio:
            j = rng.randint(0, num_items - 1)
            if j not in seen:
                samples.append((u, j, 0))
                count += 1

    return samples


def split_data(samples: list, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    # randomly split into train,val, test sets.
    rng = random.Random(seed)
    data = samples[:]
    rng.shuffle(data)
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    test = data[:n_test]
    val = data[n_test: n_test + n_val]
    train = data[n_test + n_val:]
    return train, val, test


class InteractionDataset(Dataset):
    def __init__(self, samples: list):
        users, items, labels = zip(*samples)
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]



def build_datasets(ratings_path: str, neg_ratio: int = 4, seed: int = 42):
    positives, user2idx, item2idx, num_users, num_items = load_ratings(ratings_path)

    #keep a copy of per-user positive sets for evaluation
    user_positives = defaultdict(set)
    for u, i in positives:
        user_positives[u].add(i)

    samples = sample_negatives(positives, num_items, neg_ratio=neg_ratio, seed=seed)
    train, val, test = split_data(samples, seed=seed)

    return (
        InteractionDataset(train),
        InteractionDataset(val),
        InteractionDataset(test),
        num_users,
        num_items,
        user2idx,
        item2idx,
        user_positives,
    )
