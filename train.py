import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import SASRec
from evaluation import evaluate_sasrec

DATA_PATH   = "data/ratings.dat"
MAX_SEQ_LEN = 50
HIDDEN_DIM  = 64
NUM_BLOCKS  = 2
NUM_HEADS   = 1
DROPOUT     = 0.5
LR          = 1e-3
BATCH_SIZE  = 128
NUM_EPOCHS  = 200
PATIENCE    = 10   # early-stopping patience in eval intervals
EVAL_EVERY  = 5    # evaluate on val every N epochs
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def bce_loss(pos_logits, neg_logits, target_ids):
    mask = (target_ids != 0).float()
    pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8)
    neg_loss = -torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-8)
    return ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)


(
    train_dataset,
    val_data,
    test_data,
    user_sequences,
    num_users,
    num_items,
    user2idx,
    item2idx,
) = build_datasets(DATA_PATH, max_seq_len=MAX_SEQ_LEN, seed=SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

print(f"Device: {DEVICE}")
print(f"Users: {num_users}  Items: {num_items}  Train sequences: {len(train_dataset)}")

model = SASRec(
    num_items=num_items,
    hidden_dim=HIDDEN_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    max_seq_len=MAX_SEQ_LEN,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))

best_val_ndcg  = 0.0
patience_counter = 0
best_state     = None

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss, num_batches = 0.0, 0

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

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    if epoch % EVAL_EVERY == 0:
        val_metrics = evaluate_sasrec(
            model, user_sequences, val_data,
            num_items, MAX_SEQ_LEN, DEVICE,
            k_list=[10, 20], split="val",
        )
        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
            f"Val NDCG@10 {val_metrics['ndcg@10']:.4f} | "
            f"Val Recall@10 {val_metrics['recall@10']:.4f} | "
            f"Val NDCG@20 {val_metrics['ndcg@20']:.4f} | "
            f"Val Recall@20 {val_metrics['recall@20']:.4f}"
        )

        if val_metrics["ndcg@10"] > best_val_ndcg:
            best_val_ndcg    = val_metrics["ndcg@10"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

test_metrics = evaluate_sasrec(
    model, user_sequences, test_data,
    num_items, MAX_SEQ_LEN, DEVICE,
    k_list=[10, 20], split="test",
)

print(" Test Results")
for key, val in sorted(test_metrics.items()):
    print(f"  {key}: {val:.4f}")
