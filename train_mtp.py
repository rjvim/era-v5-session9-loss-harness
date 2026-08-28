"""
train_mtp.py — resumable two-head training.

Head 0 predicts t+1. Head 1 predicts t+2, from the SAME hidden state.
Both heads are trained jointly on loss = L1 + L2.

Runs at most --chunk steps per invocation, then checkpoints and exits, so the
whole run can be driven from a sequence of short calls. State lives in
mtp_state.pt; the loss history accumulates in mtp_hist.json.

    python3 train_mtp.py --total 300 --chunk 100
"""

import argparse
import json
import math
import os
import time

import torch

from nanolm import Config, NanoLM, masked_cross_entropy
from run_all import B, T, batches, corpus_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "mtp_state.pt")
HIST = os.path.join(HERE, "mtp_hist.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(1337)

    cfg = Config(block_size=T, n_heads_out=2, tie_weights=True)
    model = NanoLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    step0 = 0
    hist = []
    if os.path.exists(STATE):
        s = torch.load(STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step0 = s["step"]
        hist = json.load(open(HIST))

    # Re-create the stream and fast-forward so resuming sees the same data order
    # it would have seen in one continuous run.
    gen = batches(corpus_tokens(), B, T, seed=11)
    for _ in range(step0):
        next(gen)

    end = min(step0 + a.chunk, a.total)
    t0 = time.time()

    for step in range(step0 + 1, end + 1):
        seq = next(gen)[:, :T + 2]
        inp = seq[:, :T]
        tgt1 = seq[:, 1:T + 1]          # t+1
        tgt2 = seq[:, 2:T + 2]          # t+2

        h = model(inp)
        l1, _ = masked_cross_entropy(model.head(h, 0), tgt1)
        l2, _ = masked_cross_entropy(model.head(h, 1), tgt2)
        loss = l1 + l2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            f1, f2 = float(l1.detach()), float(l2.detach())
            hist.append({"step": step, "l1": round(f1, 4), "l2": round(f2, 4),
                         "sum": round(f1 + f2, 4), "gap": round(f2 - f1, 4)})
            print(f"  step {step:4d}  l1 {f1:.4f}  l2 {f2:.4f}  "
                  f"gap {f2-f1:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": end}, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    torch.save(model.state_dict(), os.path.join(HERE, "mtp_ckpt.pt.tmp"))
    os.replace(os.path.join(HERE, "mtp_ckpt.pt.tmp"),
               os.path.join(HERE, "mtp_ckpt.pt"))
    json.dump(hist, open(HIST, "w"), indent=2)
    print(f"[checkpoint] step {end}/{a.total}  ({time.time()-t0:.0f}s this chunk)")
    print("DONE" if end >= a.total else "MORE")


if __name__ == "__main__":
    main()
