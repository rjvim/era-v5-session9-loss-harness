"""
nanolm.py — a deliberately small decoder-only LM plus the loss-harness pieces
that Session 9 is actually about: the shift, the masks, the output head, and
a memory-bounded cross-entropy.

Nothing here is clever. Everything here is inspectable.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 50257     # V — GPT-2 BPE vocabulary
    block_size: int = 128       # T — max context length
    n_layer: int = 4            # L — transformer blocks
    n_head: int = 4             # H — attention heads
    n_embd: int = 256           # D — model width / embedding dimension
    tie_weights: bool = True    # reuse wte as the output head
    n_heads_out: int = 1        # number of output heads (1 = ordinary, 2 = +MTP t+2)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(D, dim=2)
        # (B, H, T, d_head)
        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)

        if attn_mask is None:
            # plain causal
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attn_mask: (B, 1, T, T) boolean, True = allowed to attend.
            # Used for document-causal (block-diagonal) attention.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoLM(nn.Module):
    """
    Returns HIDDEN states, not logits. The output head is applied separately so
    that the notebook can measure the head's memory cost on its own — which is
    the entire point of items 6 and 7.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # output head(s). head i predicts token t+1+i.
        self.heads = nn.ModuleList([
            nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            for _ in range(cfg.n_heads_out)
        ])

        self.apply(self._init)
        if cfg.tie_weights:
            # only head 0 is tied; extra MTP heads stay independent
            self.heads[0].weight = self.wte.weight

    def _init(self, m):
        if isinstance(m, nn.Linear):
            # std 0.02 keeps the untrained logit distribution near-uniform,
            # which is what makes untrained perplexity land on V.
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, attn_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return self.ln_f(x)                      # (B, T, D)

    def head(self, hidden, i=0):
        return self.heads[i](hidden)             # (B, T, V)

    # -- parameter accounting -------------------------------------------
    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def n_unique_params(self):
        """Counts a tied weight once, which is what actually sits in memory."""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def shifted_targets(tokens, offset=1):
    """
    Inputs are tokens[:, :-offset]; targets are tokens[:, offset:].
    offset=1 -> next-token. offset=2 -> the MTP t+2 head.
    """
    return tokens[:, :-offset], tokens[:, offset:]


def masked_cross_entropy(logits, targets, mask=None, reduction="mean"):
    """
    logits  : (B, T, V)
    targets : (B, T)
    mask    : (B, T) bool or float, True/1 = this position contributes.

    Returns (loss, n_contributing). Materialises the full (B*T, V) logit
    tensor — this is the memory-hungry baseline for item 7.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # (B*T,)

    if mask is None:
        n = per_tok.numel()
        loss = per_tok.sum() / n
    else:
        m = mask.reshape(-1).to(per_tok.dtype)
        n = int(m.sum().item())
        loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)

    if reduction == "none":
        return per_tok, n
    return loss, n


def chunked_cross_entropy(hidden, head_weight, targets, mask=None, chunk_size=64):
    """
    Same value as masked_cross_entropy(head(hidden), targets, mask), but never
    holds more than (chunk_size, V) logits at once.

    hidden      : (B, T, D)
    head_weight : (V, D)   — nn.Linear weight, so logits = hidden @ W.T
    targets     : (B, T)
    mask        : (B, T) or None

    The trick is only that cross-entropy is a SUM over positions, so the
    positions can be visited in groups. There is no approximation here; the
    result matches the unchunked version to floating-point tolerance.
    """
    B, T, D = hidden.shape
    flat_h = hidden.reshape(-1, D)               # (N, D)
    flat_t = targets.reshape(-1)                 # (N,)
    N = flat_h.shape[0]

    if mask is None:
        flat_m = torch.ones(N, device=hidden.device, dtype=hidden.dtype)
    else:
        flat_m = mask.reshape(-1).to(hidden.dtype)

    total = hidden.new_zeros(())
    denom = flat_m.sum().clamp(min=1.0)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_c = flat_h[start:end]                  # (c, D)
        m_c = flat_m[start:end]
        if m_c.sum() == 0:
            continue
        logits_c = h_c @ head_weight.t()         # (c, V)  <- the only big tensor
        ce_c = F.cross_entropy(logits_c, flat_t[start:end], reduction="none")
        total = total + (ce_c * m_c).sum()

    return total / denom, int(flat_m.sum().item())


def chunked_cross_entropy_ckpt(hidden, head_weight, targets, mask=None, chunk_size=64):
    """
    Chunked cross-entropy WITH recomputation.

    Plain chunking (above) reduces the size of the largest live logit tensor,
    but autograd still saves every chunk's logits for the backward pass, so the
    peak barely moves. Wrapping each chunk in a checkpoint means the chunk's
    logits are thrown away after the forward and recomputed during backward.
    That is where the real saving is. The cost is one extra matmul per chunk.

    Same value and same gradients as the unchunked version.
    """
    from torch.utils.checkpoint import checkpoint

    B, T, D = hidden.shape
    flat_h = hidden.reshape(-1, D)
    flat_t = targets.reshape(-1)
    N = flat_h.shape[0]

    if mask is None:
        flat_m = torch.ones(N, device=hidden.device, dtype=hidden.dtype)
    else:
        flat_m = mask.reshape(-1).to(hidden.dtype)

    def chunk_sum(h_c, t_c, m_c, W):
        logits_c = h_c @ W.t()
        ce_c = F.cross_entropy(logits_c, t_c, reduction="none")
        return (ce_c * m_c).sum()

    total = hidden.new_zeros(())
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if flat_m[start:end].sum() == 0:
            continue
        total = total + checkpoint(
            chunk_sum,
            flat_h[start:end], flat_t[start:end], flat_m[start:end], head_weight,
            use_reentrant=False,
        )

    return total / flat_m.sum().clamp(min=1.0), int(flat_m.sum().item())


# --------------------------------------------------------------------------
# document packing / masks
# --------------------------------------------------------------------------

def build_doc_ids(lengths, device="cpu"):
    """[3, 5] -> tensor([0,0,0,1,1,1,1,1])"""
    out = []
    for i, L in enumerate(lengths):
        out.append(torch.full((L,), i, dtype=torch.long, device=device))
    return torch.cat(out)


def document_attn_mask(doc_ids):
    """
    Block-diagonal causal mask: position i may attend to j iff j <= i AND
    doc_ids[i] == doc_ids[j]. Returns (1, 1, T, T) bool for SDPA.

    Loss-masking the boundary TARGET stops the model being graded on an
    impossible prediction. It does NOT stop document B's tokens from reading
    document A through attention. That requires this.
    """
    T = doc_ids.shape[0]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=doc_ids.device))
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    return (causal & same_doc)[None, None, :, :]


def boundary_loss_mask(doc_ids, offset=1):
    """
    doc_ids : (T,) document id per position in the PACKED sequence.

    After a shift of `offset`, prediction slot i has input doc_ids[i] and
    target doc_ids[i+offset]. The slot is legitimate only when those match.
    Returns a (T-offset,) bool mask, True = counts toward the loss.

    For two documents of length a and b packed into one sequence, exactly
    `offset` slots get dropped.
    """
    return doc_ids[:-offset] == doc_ids[offset:]
