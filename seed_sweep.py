"""
seed_sweep.py — is the Part 2 result a property of the setup, or of seed 1337?

Trains the same two-head model under additional seeds and checks whether the
claims that carry the write-up survive:

  * the step-1 gap sign is noise (not reliably negative)
  * head 1 crosses above head 0 early and stays there
  * the early gap slope is strongly positive

Resumable, same pattern as train_mtp.py:
    python3 seed_sweep.py --seed 7 --total 300 --chunk 70
"""

import argparse
import json
import os
import time

import torch

from nanolm import Config, NanoLM, masked_cross_entropy
from run_all import B, T, batches, corpus_tokens

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=70)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    state = os.path.join(HERE, f"seed{a.seed}_state.pt")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)

    cfg = Config(block_size=T, n_heads_out=2, tie_weights=True)
    model = NanoLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    step0, hist = 0, []
    if os.path.exists(state):
        s = torch.load(state, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step0, hist = s["step"], s["hist"]

    # data order also varies with the seed -- otherwise this only tests init
    gen = batches(corpus_tokens(), B, T, seed=a.seed)
    for _ in range(step0):
        next(gen)

    end = min(step0 + a.chunk, a.total)
    t0 = time.time()
    for step in range(step0 + 1, end + 1):
        seq = next(gen)[:, :T + 2]
        h = model(seq[:, :T])
        l1, _ = masked_cross_entropy(model.head(h, 0), seq[:, 1:T + 1])
        l2, _ = masked_cross_entropy(model.head(h, 1), seq[:, 2:T + 2])
        loss = l1 + l2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0 or step == 1:
            f1, f2 = float(l1.detach()), float(l2.detach())
            hist.append({"step": step, "l1": round(f1, 4), "l2": round(f2, 4),
                         "gap": round(f2 - f1, 4)})
            print(f"  seed {a.seed} step {step:4d}  l1 {f1:.4f}  l2 {f2:.4f}  "
                  f"gap {f2-f1:+.4f}", flush=True)

    # atomic: write to a temp file then rename, so a kill mid-write cannot
    # leave a truncated checkpoint behind (it did, once).
    tmp = state + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": end, "hist": hist}, tmp)
    os.replace(tmp, state)
    json.dump(hist, open(os.path.join(HERE, f"seed{a.seed}_hist.json"), "w"), indent=1)
    print(f"[seed {a.seed}] {end}/{a.total} "
          f"{'DONE' if end >= a.total else 'MORE'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
