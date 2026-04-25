import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # (batch, seq, dim) convention
        )
        self.ffn = FeedForward(hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # self attention with mask
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask, # (seq_len, seq_len) additive mask
            key_padding_mask=key_padding_mask,  # (batch, seq_len) True = ignore
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        # position-wise feedforward
        x = x + self.ffn(self.norm2(x))
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        max_seq_len: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # item embedding
        self.item_emb = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        # positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # layer norm applied to the output of the last block
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # Keep the padding embedding exactly at zero
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_seq.shape

        # positional indices
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)

        x = self.item_emb(input_seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_seq.device),
            diagonal=1,
        )

        # key padding mask
        key_padding_mask = (input_seq == 0)  # (batch, seq_len)

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)

        return self.norm(x)  # (batch, seq_len, hidden_dim)

    def forward(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):

        seq_repr = self._encode(input_seq)

        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)

        pos_logits = (seq_repr * pos_emb).sum(-1)
        neg_logits = (seq_repr * neg_emb).sum(-1)

        return pos_logits, neg_logits

    def predict_scores(self, input_seq: torch.Tensor) -> torch.Tensor:

        seq_repr = self._encode(input_seq)

        lengths = (input_seq != 0).sum(dim=1).clamp(min=1)  # (batch,)
        batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)
        last_repr = seq_repr[batch_idx, lengths - 1]  # (batch, dim)

        all_emb = self.item_emb.weight                # (num_items, dim)
        return last_repr @ all_emb.T                  # (batch, num_items)
