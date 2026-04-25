import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import SASRec
from evaluation import evaluate_sasrec

DATA_PATH  = "data/ratings.dat"
STUDY_NAME = "sasrec_study"
DB_URL     = "sqlite:///study.sqlite3"
N_TRIALS   = 6000
SEED       = 42
MAX_EPOCHS = 100
EVAL_EVERY = 5     # evaluate val every N epochs
PATIENCE   = 5     # early-stop patience

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_dataset_cache: dict = {}

def get_datasets(max_seq_len: int):
    if max_seq_len not in _dataset_cache:
        _dataset_cache[max_seq_len] = build_datasets(
            DATA_PATH, max_seq_len=max_seq_len, seed=SEED
        )
    return _dataset_cache[max_seq_len]

def bce_loss(pos_logits, neg_logits, target_ids):
    mask = (target_ids != 0).float()
    pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8)
    neg_loss = -torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-8)
    return ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)


def objective(trial: optuna.Trial) -> float:
    hidden_dim  = trial.suggest_categorical("hidden_dim",  [32, 64, 128, 256])
    num_heads   = trial.suggest_categorical("num_heads",   [1, 2, 4, 8])
    num_blocks  = trial.suggest_int("num_blocks",  1, 4)
    max_seq_len = trial.suggest_categorical("max_seq_len", [50, 100, 200])
    dropout     = trial.suggest_float("dropout",   0.1, 0.6, step=0.1)
    lr          = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size",  [64, 128, 256])

    (train_dataset, val_data, _test_data,
     user_sequences, _num_users, num_items,
     _user2idx, _item2idx) = get_datasets(max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model = SASRec(
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    best_val_ndcg    = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for input_seq, pos_items, neg_items in train_loader:
            input_seq = input_seq.to(DEVICE)
            pos_items = pos_items.to(DEVICE)
            neg_items = neg_items.to(DEVICE)

            pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
            loss = bce_loss(pos_logits, neg_logits, pos_items)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch % EVAL_EVERY == 0:
            val_metrics = evaluate_sasrec(
                model, user_sequences, val_data,
                num_items, max_seq_len, DEVICE,
                k_list=[10], split="val",
            )
            val_ndcg10 = val_metrics["ndcg@10"]

            # Let Optuna prune unpromising trials early
            trial.report(val_ndcg10, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_ndcg10 > best_val_ndcg:
                best_val_ndcg    = val_ndcg10
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

    return best_val_ndcg

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=DB_URL,
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    load_if_exists=True,   # safe to re-run; resumes from where it left off
)

print(f"Device : {DEVICE}")
print(f"Storage: {DB_URL}  (study: {STUDY_NAME})")
print(f"Trials : {N_TRIALS}  (already done: {len(study.trials)})")
print()

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n=== Best Trial ===")
best = study.best_trial
print(f"  Val NDCG@10 : {best.value:.4f}")
print(f"  Params:")
for k, v in best.params.items():
    print(f"    {k}: {v}")

print(f"\nTo inspect results run:")
print(f"  optuna-dashboard {DB_URL}")
