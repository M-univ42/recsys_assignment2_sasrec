import torch
import numpy as np

def recall_at_k(ranked_items, ground_truth: set, k: int) -> float:
    hits = sum(1 for item in ranked_items[:k] if item in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(ranked_items, ground_truth: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(ranked_items[:k])
        if item in ground_truth
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_sasrec(
    model,
    user_sequences: dict,
    eval_data: dict,
    num_items: int,
    max_seq_len: int,
    device: str,
    k_list: list = None,
    split: str = "val",
) -> dict:
    """
    Full-ranking evaluation for SASRec.

    For each user the model receives the sequence prefix as input and scores
    every item in the catalogue.  Training items are masked to -inf so they
    cannot appear in the ranked list.

    Parameters
    ----------
    model        : trained SASRec instance
    user_sequences : full sequences (including val/test items) keyed by user idx
    eval_data    : {user_idx: target_item_idx}  (val or test split)
    num_items    : total number of items (including padding at index 0)
    max_seq_len  : maximum sequence length used during training
    device       : 'cpu' or 'cuda'
    k_list       : list of cut-off values, default [10, 20]
    split        : 'val' uses seq[:-2] as prefix; 'test' uses seq[:-1]
    """
    if k_list is None:
        k_list = [10, 20]

    model.eval()
    metrics = {f"recall@{k}": [] for k in k_list}
    metrics.update({f"ndcg@{k}": [] for k in k_list})

    # Number of items to exclude from the tail of the sequence
    exclude_tail = 2 if split == "val" else 1

    with torch.no_grad():
        for uid, target_item in eval_data.items():
            seq = user_sequences.get(uid, [])
            if len(seq) < (exclude_tail + 1):
                continue

            # Prefix used as model input (excludes val+test for val; excludes test for test)
            prefix = seq[:-exclude_tail]

            # Left-pad to max_seq_len
            padded = prefix[-max_seq_len:]
            padded = [0] * (max_seq_len - len(padded)) + padded

            seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            scores = model.predict_scores(seq_tensor).squeeze(0).cpu().numpy()

            # Mask padding index and all items seen during training
            train_items = set(seq[:-2])   # always mask training items regardless of split
            scores[0] = -np.inf
            for item in train_items:
                scores[item] = -np.inf

            ranked = np.argsort(-scores)
            ground_truth = {target_item}

            for k in k_list:
                metrics[f"recall@{k}"].append(recall_at_k(ranked, ground_truth, k))
                metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, ground_truth, k))

    return {
        key: float(np.mean(vals)) if vals else 0.0
        for key, vals in metrics.items()
    }
