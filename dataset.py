import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def load_ratings(path: str, min_rating: int = 4, min_seq_len: int = 5):
    raw = []
    user_seq = {}

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            uid, iid, rating, timestamp = (
                int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
            )
            if rating >= min_rating:
                raw.append((uid, iid, timestamp))

    for uid, iid, timestamp in raw:
        if uid not in user_seq:
            user_seq[uid] = []
        user_seq[uid].append((iid, timestamp))

    # keep only users with at least min_seq_len positive interactions
    user_seq = {uid: seq for uid, seq in user_seq.items() if len(seq) >= min_seq_len}

    # map original IDs to sequential indices; item 0 is reserved for padding
    user2idx = {u: i for i, u in enumerate(sorted(user_seq.keys()))}
    item_set = set(iid for seq in user_seq.values() for iid, _ in seq)
    item2idx = {it: i + 1 for i, it in enumerate(sorted(item_set))}  # 1-indexed

    num_users = len(user2idx)
    num_items = len(item2idx) + 1  # +1 so index 0 stays as padding

    # sort each users interactions by timestamp and convert to item indices
    user_sequences = {}
    for uid, interactions in user_seq.items():
        interactions.sort(key=lambda x: x[1])
        user_sequences[user2idx[uid]] = [item2idx[iid] for iid, _ in interactions]

    return user_sequences, user2idx, item2idx, num_users, num_items


class SeqDataset(Dataset):
    def __init__(self, user_sequences: dict, num_items: int, max_seq_len: int = 50, seed: int = 42):
        self.max_len = max_seq_len
        self.num_items = num_items
        self.rng = random.Random(seed)

        self.samples = [] # list of (uid, train_seq)
        self.user_item_sets = {}

        for uid, seq in user_sequences.items():
            self.user_item_sets[uid] = set(seq)
            # last item = test, second-to-last = val
            train_seq = seq[:-2]
            if len(train_seq) < 2:  # need at least one pair
                continue
            self.samples.append((uid, train_seq))

    def __len__(self):
        return len(self.samples)

    def _pad(self, seq: list) -> list:
        seq = seq[-self.max_len:]
        return [0] * (self.max_len - len(seq)) + seq

    def _sample_neg(self, uid: int) -> int:
        seen = self.user_item_sets[uid]
        while True:
            neg = self.rng.randint(1, self.num_items - 1)
            if neg not in seen:
                return neg

    def __getitem__(self, idx: int):
        uid, train_seq = self.samples[idx]
        input_ids = train_seq[:-1]   # prefix
        target_ids = train_seq[1:]   # next items
        neg_ids = [self._sample_neg(uid) for _ in target_ids]

        return (
            torch.tensor(self._pad(input_ids),  dtype=torch.long),
            torch.tensor(self._pad(target_ids), dtype=torch.long),
            torch.tensor(self._pad(neg_ids),    dtype=torch.long),
        )


def build_datasets(ratings_path: str, max_seq_len: int = 50, seed: int = 42):
    user_sequences, user2idx, item2idx, num_users, num_items = load_ratings(ratings_path)

    #keep a copy of per-user positive sets for evaluation
    val_data  = {u: seq[-2] for u, seq in user_sequences.items()}
    test_data = {u: seq[-1] for u, seq in user_sequences.items()}

    train_dataset = SeqDataset(user_sequences, num_items, max_seq_len=max_seq_len, seed=seed)

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
