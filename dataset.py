import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

def load_ratings(path: str, min_rating: int = 4):
    # user_set, item_set = set(), set()
    raw = []
    user_set = {}
    user_seq = {}

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            uid, iid, rating, timestamp = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
            if rating >= min_rating:
                raw.append((uid, iid, timestamp))
                # user_set.add(uid)
                # item_set.add(iid)

    # user2idx = {u: i for i, u in enumerate(sorted(user_set))}
    # item2idx = {it: i for i, it in enumerate(sorted(item_set))}

    # positives = [(user2idx[u], item2idx[i]) for u, i in raw_positives]
    # return positives, user2idx, item2idx, len(user2idx), len(item2idx)

    for uid, iid, timestamp in raw:
        if uid not in user_set:
            user_set[uid] = len(user_set)
            user_seq[uid] = []
        user_seq[uid].append((iid, timestamp))
    user_seq = {uid: seq for uid, seq in user_seq.items() if len(seq) >= 5}
    user2idx = {u: i for i, u in enumerate(sorted(user_set.keys()))}
    item_set = set(iid for seq in user_seq.values() for iid, _ in seq)
    item2idx = {it: i+1 for i, it in enumerate(sorted(item_set))}
    no_users = len(user2idx)
    no_items = len(item2idx)+1
    user_sequences = {}
    for uid, interactions in user_seq.items():
        interactions.sort(key=lambda x: x[1])
        user_sequences[user2idx[uid]] = [item2idx[iid] for iid, _ in interactions]
    return user_seq, user2idx, item2idx, no_users, no_items



# def sample_negatives(positives: list, num_items: int, neg_ratio: int = 4, seed: int = 42):
#     rng = random.Random(seed)
#     user_positives = defaultdict(set)
#     for u, i in positives:
#         user_positives[u].add(i)

#     all_items = list(range(num_items))
#     samples = []

#     for u, i in positives:
#         samples.append((u, i, 1))
#         seen = user_positives[u]
#         count = 0
#         while count < neg_ratio:
#             j = rng.randint(0, num_items - 1)
#             if j not in seen:
#                 samples.append((u, j, 0))
#                 count += 1

#     return samples


# def split_data(samples: list, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
#     # randomly split into train,val, test sets.
#     rng = random.Random(seed)
#     data = samples[:]
#     rng.shuffle(data)
#     n = len(data)
#     n_val = int(n * val_ratio)
#     n_test = int(n * test_ratio)
#     test = data[:n_test]
#     val = data[n_test: n_test + n_val]
#     train = data[n_test + n_val:]
#     return train, val, test


# class InteractionDataset(Dataset):
#     def __init__(self, samples: list):
#         users, items, labels = zip(*samples)
#         self.users = torch.tensor(users, dtype=torch.long)
#         self.items = torch.tensor(items, dtype=torch.long)
#         self.labels = torch.tensor(labels, dtype=torch.float32)

#     def __len__(self):
#         return len(self.users)

#     def __getitem__(self, idx):
#         return self.users[idx], self.items[idx], self.labels[idx]

class SeqDataset(Dataset):
    def __init__(self, user_seq, no_items,user2idx, item2idx, max_seq_len=50,seed=42):
        self.max_len = max_seq_len
        self.num_items = no_items
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.users)
    
    def padding(self, seq):
        seq = seq[-self.max_len:] 
        pad_len = self.max_len - len(seq)
        return [0] * pad_len + seq
        
    def sample_neg(self,user):
        seen = self.user_item_sets[user]
        while True:
            neg = self.rng.randint(1, self.num_items - 1)
            if neg not in seen:
                return neg
    def __getitem__(self, id):
        user, input_seq, target = self.samples[id]
        padded = self.padding(input_seq)
        neg = self._sample_negative(user)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )
    

def build_datasets(ratings_path: str, max_seq_len: int,neg_ratio: int = 4, seed: int = 42):
    user_sequences, user2idx, item2idx, num_users, num_items = load_ratings(ratings_path)

    #keep a copy of per-user positive sets for evaluation
    user_positives = defaultdict(set)
    val_data  = {u: seq[-2] for u, seq in user_sequences.items()} 
    test_data = {u: seq[-1] for u, seq in user_sequences.items()} 
    train_dataset = SeqDataset(user_sequences, num_items, max_len=max_seq_len, seed=seed)
    return (
        train_dataset,
        val_data,        
        test_data,       
        user_sequences,  
        num_users,
        num_items,
        user2idx,
        item2idx,
    )
