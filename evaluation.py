import torch
import numpy as np


def recall_at_k(ranked_items, ground_truth, k):
    top_k = ranked_items[:k]
    hits = len(set(top_k) & ground_truth)
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0


def ndcg_at_k(ranked_items, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(model, user_positives, num_users, num_items, k=10, device="cpu", train_positives=None):
    model.eval()
    recalls = []
    ndcgs = []

    if train_positives is None:
        train_positives = {}

    with torch.no_grad():
        for user in range(num_users):
            if user not in user_positives:
                continue

            items = torch.arange(num_items).to(device)
            users = torch.full((num_items,), user).to(device)

            scores = model(users, items)
            scores = scores.cpu().numpy()

            seen_items = train_positives.get(user, set())
            if len(seen_items) > 0:
                scores[list(seen_items)] = -np.inf

            ranked_items = np.argsort(-scores)
            gt_items = user_positives[user]

            recalls.append(recall_at_k(ranked_items, gt_items, k))
            ndcgs.append(ndcg_at_k(ranked_items, gt_items, k))

    return {
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }