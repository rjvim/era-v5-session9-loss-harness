"""
mem_probe.py — peak memory for full vs chunked cross-entropy.

IMPORTANT: run this with

    MALLOC_MMAP_THRESHOLD_=65536 MALLOC_TRIM_THRESHOLD_=65536

glibc's malloc caches freed blocks instead of returning them to the OS, so peak
RSS over-reports: it measures the high-water mark of the ALLOCATOR, not of live
memory. Forcing allocations above 64 KB through mmap makes them unmap on free.
Without this the chunked+checkpoint row varies by ~330 MB between runs; with it
the spread is under 1 MB. run_all.py sets these in the subprocess environment.

Why a subprocess: on CPU there is no torch.cuda.max_memory_allocated(). The
honest CPU equivalent is the peak resident set size of the process, and
ru_maxrss is a high-water mark that never resets. So each variant is measured
in a FRESH process and compared against a baseline process that builds the same
model and runs the same forward pass but computes no loss at all.

    peak_attributable = peak_variant - peak_baseline

Run:  python3 mem_probe.py <mode> <B> <T> <chunk>
      mode in {baseline, full, chunked}
Prints one JSON line.
"""

import json
import os
import resource
import sys
import time

import torch

from nanolm import (Config, NanoLM, chunked_cross_entropy, chunked_cross_entropy_ckpt,
                    masked_cross_entropy, shifted_targets)


def peak_rss_mb():
    # Linux reports ru_maxrss in kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_mb(device):
    """
    On CUDA this is exact and deterministic. On CPU it is peak RSS, which is a
    high-water mark of the whole process and is contaminated by allocator
    caching -- treat it as a proxy and always report a spread.
    """
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6, "cuda.max_memory_allocated"
    return peak_rss_mb(), "peak_rss"


def saved_bytes(fn):
    """
    Exact bytes retained by the autograd graph, counted by intercepting every
    tensor the backward pass saves. Deterministic and machine-independent --
    unlike peak RSS, this number is identical on any box.
    """
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = fn()
    return total, loss


def main():
    mode = sys.argv[1]
    B, T, chunk = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cfg = Config(block_size=T)
    model = NanoLM(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    inp, tgt = shifted_targets(tokens)

    loss_val = None
    t0 = time.perf_counter()

    if mode == "baseline":
        # forward through the trunk only — no head, no logits, no loss.
        h = model(inp)
        h.sum().backward()

    elif mode == "full":
        h = model(inp)
        logits = model.head(h)                       # (B, T-1, V) materialised
        loss, _ = masked_cross_entropy(logits, tgt)
        loss.backward()
        loss_val = float(loss.detach())

    elif mode == "chunked":
        h = model(inp)
        loss, _ = chunked_cross_entropy(
            h, model.heads[0].weight, tgt, chunk_size=chunk
        )
        loss.backward()
        loss_val = float(loss.detach())

    elif mode.startswith("saved_"):
        h = model(inp)
        which = mode.split("_", 1)[1]
        if which == "full":
            nb, loss = saved_bytes(
                lambda: masked_cross_entropy(model.head(h), tgt)[0])
        elif which == "chunked":
            nb, loss = saved_bytes(
                lambda: chunked_cross_entropy(
                    h, model.heads[0].weight, tgt, chunk_size=chunk)[0])
        else:
            nb, loss = saved_bytes(
                lambda: chunked_cross_entropy_ckpt(
                    h, model.heads[0].weight, tgt, chunk_size=chunk)[0])
        loss_val = float(loss.detach())
        print(json.dumps({"mode": mode, "B": B, "T": T, "chunk": chunk,
                          "saved_mb": round(nb / 1e6, 2), "loss": loss_val}))
        return

    elif mode == "chunked_ckpt":
        h = model(inp)
        loss, _ = chunked_cross_entropy_ckpt(
            h, model.heads[0].weight, tgt, chunk_size=chunk
        )
        loss.backward()
        loss_val = float(loss.detach())

    else:
        raise SystemExit(f"unknown mode {mode}")

    if device == "cuda":
        torch.cuda.synchronize()
    pk, instrument = peak_mb(device)
    print(json.dumps({
        "mode": mode, "B": B, "T": T, "chunk": chunk,
        "device": device, "instrument": instrument,
        "mmap_forced": os.environ.get("MALLOC_MMAP_THRESHOLD_") is not None,
        "peak_rss_mb": round(pk, 1),
        "sec": round(time.perf_counter() - t0, 3),
        "loss": loss_val,
    }))


if __name__ == "__main__":
    main()
