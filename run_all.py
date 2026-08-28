"""
run_all.py — produces every number claimed in the README, into results.json.

Stages are cached, so a stage can be re-run in isolation:
    python3 run_all.py shapes shift pad boundary ppl params memory mtp shiftbug
    python3 run_all.py all
"""

import json
import math
import os
import statistics
import subprocess
import sys
import time

import tiktoken
import torch
import torch.nn.functional as F

from nanolm import (
    Config, NanoLM, boundary_loss_mask, build_doc_ids, chunked_cross_entropy,
    chunked_cross_entropy_ckpt, document_attn_mask, masked_cross_entropy,
    shifted_targets,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENC = tiktoken.get_encoding("gpt2")
PAD_ID = ENC.n_vocab - 1        # reuse the last slot as PAD (no dedicated pad in GPT-2 BPE)

B, T = 8, 128
MEM_B, MEM_T = 8, 256


def load(): return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
def save(d): json.dump(d, open(RESULTS, "w"), indent=2)


CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
              "master/data/tinyshakespeare/input.txt")


def corpus_path():
    p = os.path.join(HERE, "input.txt")
    if not os.path.exists(p):
        import urllib.request
        print("downloading TinyShakespeare ...")
        urllib.request.urlretrieve(CORPUS_URL, p)
    return p


def corpus_tokens():
    text = open(corpus_path()).read()
    return torch.tensor(ENC.encode(text), dtype=torch.long)


def batches(data, bs, bt, seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(len(data) - bt - 2, (bs,), generator=g)
        yield torch.stack([data[i:i + bt + 2] for i in ix])


# ==========================================================================
# 1. shapes
# ==========================================================================
def stage_shapes(R):
    torch.manual_seed(0)
    cfg = Config(block_size=T)
    m = NanoLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    inp, tgt = shifted_targets(tokens)
    h = m(inp)
    logits = m.head(h)
    loss, n = masked_cross_entropy(logits, tgt)

    rows = [
        ("tokens", tuple(tokens.shape), "B=batch, T=packed sequence length"),
        ("inputs  = tokens[:, :-1]", tuple(inp.shape), "B, T-1 positions the model reads"),
        ("targets = tokens[:, 1:]", tuple(tgt.shape), "B, T-1 next-token answers"),
        ("hidden", tuple(h.shape), "B, T-1, D=model width"),
        ("logits", tuple(logits.shape), "B, T-1, V=vocabulary size"),
        ("flat logits", (logits.shape[0] * logits.shape[1], logits.shape[2]),
         "B*(T-1) rows, one V-way classification each"),
        ("flat targets", (tgt.numel(),), "B*(T-1) class indices"),
        ("loss", tuple(loss.shape), "scalar — mean over contributing positions"),
    ]
    R["shapes"] = {
        "rows": [{"tensor": a, "shape": list(b), "meaning": c} for a, b, c in rows],
        "B": B, "T": T, "D": cfg.n_embd, "V": cfg.vocab_size,
        "logit_bytes_fp32_mb": round(logits.numel() * 4 / 1e6, 1),
        "contributing_tokens": n,
    }
    return R


# ==========================================================================
# 2. verify the shift by printing STRINGS
# ==========================================================================
def stage_shift(R):
    text = "The capital of India is New Delhi and the capital of France is Paris"
    ids = torch.tensor(ENC.encode(text))[None, :]
    inp, tgt = shifted_targets(ids, 1)

    correct = [{"i": i,
                "input": ENC.decode([int(inp[0, i])]),
                "target": ENC.decode([int(tgt[0, i])])}
               for i in range(inp.shape[1])]

    # the bug the assignment warns about: shifting the wrong way, so the model
    # is asked to predict the token it has just been shown.
    wrong_inp, wrong_tgt = ids[:, 1:], ids[:, :-1]
    wrong = [{"i": i,
              "input": ENC.decode([int(wrong_inp[0, i])]),
              "target": ENC.decode([int(wrong_tgt[0, i])])}
             for i in range(wrong_inp.shape[1])]

    # a no-shift version: target == input, the most seductive bug of all
    noshift = [{"i": i,
                "input": ENC.decode([int(ids[0, i])]),
                "target": ENC.decode([int(ids[0, i])])}
               for i in range(ids.shape[1])]

    R["shift"] = {
        "text": text,
        "correct": correct,
        "reversed": wrong,
        "no_shift": noshift,
        "correct_is_next_token": all(
            correct[i]["target"] == correct[i + 1]["input"]
            for i in range(len(correct) - 1)
        ),
    }
    return R


# ==========================================================================
# 3. padding mask
# ==========================================================================
def stage_pad(R):
    torch.manual_seed(0)
    cfg = Config(block_size=T)
    m = NanoLM(cfg)

    real_len = 40
    seq = torch.randint(0, 50000, (B, T))
    seq[:, real_len:] = PAD_ID                    # everything after real_len is PAD
    inp, tgt = shifted_targets(seq)

    h = m(inp)
    logits = m.head(h)

    loss_unmasked, n_unmasked = masked_cross_entropy(logits, tgt)
    mask = (tgt != PAD_ID)
    loss_masked, n_masked = masked_cross_entropy(logits, tgt, mask=mask)

    # ---- the part that actually matters: what happens once you TRAIN ----
    # On an untrained model, masking barely moves the loss, because every
    # prediction is equally hopeless. The damage shows up after training: PAD
    # is trivially predictable, so an unmasked loss is dominated by free wins.
    def short_train(use_mask, steps=60, lr=3e-4):
        torch.manual_seed(1337)
        c = Config(block_size=T)
        mm = NanoLM(c)
        opt = torch.optim.AdamW(mm.parameters(), lr=lr, weight_decay=0.01)
        data = corpus_tokens()
        gen = batches(data, B, T, seed=5)
        hist = []
        for step in range(1, steps + 1):
            real = next(gen)[:, :T + 1].clone()
            real[:, real_len:] = PAD_ID          # same padding regime
            i2, t2 = real[:, :T], real[:, 1:T + 1]
            lg = mm.head(mm(i2))
            msk = (t2 != PAD_ID) if use_mask else None
            l, _ = masked_cross_entropy(lg, t2, mask=msk)
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == 1:
                hist.append({"step": step, "reported_loss": round(float(l), 4)})
        # honest evaluation: loss on REAL tokens only, identical both ways
        with torch.no_grad():
            ev = next(batches(data, B, T, seed=808))[:, :T + 1].clone()
            ev[:, real_len:] = PAD_ID
            lg = mm.head(mm(ev[:, :T]))
            t2 = ev[:, 1:T + 1]
            content, _ = masked_cross_entropy(lg, t2, mask=(t2 != PAD_ID))
        return hist, round(float(content), 4)

    PAD_TRAIN_STEPS = 60
    h_un, content_un = short_train(use_mask=False, steps=PAD_TRAIN_STEPS)
    h_ma, content_ma = short_train(use_mask=True, steps=PAD_TRAIN_STEPS)

    R["pad"] = {
        "pad_id": PAD_ID, "real_len": real_len, "B": B, "T": T,
        "n_unmasked": n_unmasked, "n_masked": n_masked,
        "n_dropped": n_unmasked - n_masked,
        "train_steps": PAD_TRAIN_STEPS,
        "pct_of_slots_that_were_pad": round(100 * (n_unmasked - n_masked) / n_unmasked, 1),
        "untrained": {
            "loss_unmasked": round(float(loss_unmasked), 4),
            "loss_masked": round(float(loss_masked), 4),
            "ppl_unmasked": round(math.exp(float(loss_unmasked)), 1),
            "ppl_masked": round(math.exp(float(loss_masked)), 1),
            "delta": round(float(loss_masked) - float(loss_unmasked), 4),
        },
        "trained_short_run": {
            "unmasked_run": {"history": h_un,
                             "reported_loss_final": h_un[-1]["reported_loss"],
                             "true_content_loss": content_un},
            "masked_run": {"history": h_ma,
                           "reported_loss_final": h_ma[-1]["reported_loss"],
                           "true_content_loss": content_ma},
            "reported_loss_gap": round(h_ma[-1]["reported_loss"]
                                       - h_un[-1]["reported_loss"], 4),
            "true_content_loss_gap": round(content_un - content_ma, 4),
        },
    }
    return R


# ==========================================================================
# 5. untrained perplexity vs vocabulary size
# ==========================================================================
def stage_ppl(R):
    """
    An untrained model should sit at perplexity ~= V, because a uniform guess
    over V classes costs ln(V) nats. It never lands exactly on V, and the
    residual is not noise -- it is predictable. For logits that are independent
    of the target with variance s2,

        E[CE] = ln(V) + s2/2

    Three conditions isolate where the residual comes from.
    """
    lnV = math.log(Config().vocab_size)

    def measure(tie, random_tokens, seed=0):
        torch.manual_seed(seed)
        cfg = Config(block_size=T, tie_weights=tie)
        m = NanoLM(cfg)
        if random_tokens:
            tokens = torch.randint(0, cfg.vocab_size, (B, T),
                                   generator=torch.Generator().manual_seed(3))
        else:
            tokens = next(batches(corpus_tokens(), B, T, seed=7))[:, :T]
        inp, tgt = shifted_targets(tokens)
        with torch.no_grad():
            logits = m.head(m(inp))
            loss, _ = masked_cross_entropy(logits, tgt)
        L, s2 = float(loss), float(logits.var())
        return {
            "loss": round(L, 4),
            "ppl": round(math.exp(L), 1),
            "ratio_ppl_over_V": round(math.exp(L) / cfg.vocab_size, 4),
            "logit_variance": round(s2, 5),
            "predicted_loss": round(lnV + s2 / 2, 4),
            "prediction_error_nats": round(L - (lnV + s2 / 2), 6),
        }

    untied_rand = measure(tie=False, random_tokens=True)
    tied_rand = measure(tie=True, random_tokens=True)
    tied_real = measure(tie=True, random_tokens=False)

    R["ppl"] = {
        "V": Config().vocab_size,
        "ln_V": round(lnV, 4),
        # the headline number the assignment asks for
        "loss_untrained": tied_real["loss"],
        "ppl_untrained": tied_real["ppl"],
        "ratio_ppl_over_V": tied_real["ratio_ppl_over_V"],
        "conditions": {
            "untied_head_random_tokens": untied_rand,
            "tied_head_random_tokens": tied_rand,
            "tied_head_real_text": tied_real,
        },
        # if the formula is exact in the control condition, it is the tying and
        # the token statistics -- not a bug -- that move the real-text number.
        "formula_exact_in_control": abs(untied_rand["prediction_error_nats"]) < 1e-2,
    }
    return R


# ==========================================================================
# 6. tied vs untied parameter counts
# ==========================================================================
def stage_params(R):
    cfg_t = Config(block_size=T, tie_weights=True)
    cfg_u = Config(block_size=T, tie_weights=False)
    mt, mu = NanoLM(cfg_t), NanoLM(cfg_u)

    emb = cfg_t.vocab_size * cfg_t.n_embd
    tied, untied = mt.n_unique_params(), mu.n_unique_params()

    R["params"] = {
        "V": cfg_t.vocab_size, "D": cfg_t.n_embd,
        "n_layer": cfg_t.n_layer, "n_head": cfg_t.n_head,
        "embedding_matrix_params": emb,
        "tied_total": tied,
        "untied_total": untied,
        "difference": untied - tied,
        "difference_equals_embedding_matrix": (untied - tied) == emb,
        "tied_pct_in_embedding": round(100 * emb / tied, 1),
        "untied_pct_in_embedding": round(100 * 2 * emb / untied, 1),
    }
    return R


# ==========================================================================
# 7. memory: full vs chunked cross-entropy
# ==========================================================================
MALLOC_ENV = {"MALLOC_MMAP_THRESHOLD_": "65536", "MALLOC_TRIM_THRESHOLD_": "65536"}


def stage_memory(R, reps=7, chunk=64):
    """
    Two instruments, because one of them is a proxy and the other is exact.

    peak RSS  -- what the process actually needed. Only meaningful with
                 MALLOC_MMAP_THRESHOLD_ set; glibc otherwise caches freed
                 blocks and the measurement drifts by hundreds of MB.
    saved bytes -- exactly what the autograd graph retains, counted by
                 intercepting every saved tensor. Identical on any machine.
    """
    env = dict(os.environ, **MALLOC_ENV)

    def probe(mode):
        vals, secs, losses, saved = [], [], [], []
        for _ in range(reps):
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "mem_probe.py"),
                 mode, str(MEM_B), str(MEM_T), str(chunk)],
                capture_output=True, text=True, check=True, env=env,
            )
            d = json.loads(out.stdout.strip().splitlines()[-1])
            if "saved_mb" in d:
                saved.append(d["saved_mb"]); losses.append(d["loss"]); continue
            vals.append(d["peak_rss_mb"]); secs.append(d["sec"]); losses.append(d["loss"])
        if saved:
            return {"saved_mb": saved[0], "deterministic": len(set(saved)) == 1,
                    "loss": losses[0]}
        return {
            "median_mb": statistics.median(vals),
            "min_mb": min(vals), "max_mb": max(vals),
            "spread_mb": round(max(vals) - min(vals), 2),
            "median_sec": round(statistics.median(secs), 3),
            "loss": losses[0], "instrument": d["instrument"],
            "device": d["device"], "mmap_forced": d["mmap_forced"],
        }

    base = probe("baseline")
    full = probe("full")
    chk = probe("chunked")
    ckpt = probe("chunked_ckpt")
    s_full = probe("saved_full")
    s_chk = probe("saved_chunked")
    s_ckpt = probe("saved_ckpt")

    def attr(x): return round(x["median_mb"] - base["median_mb"], 1)

    V, D = Config().vocab_size, Config().n_embd
    n_pos = MEM_B * (MEM_T - 1)
    analytic_full = n_pos * V * 4 / 1e6
    analytic_chunk = chunk * V * 4 / 1e6

    R["memory"] = {
        "config": {"B": MEM_B, "T": MEM_T, "V": V, "D": D, "chunk": chunk,
                   "reps": reps},
        "instrument": full["instrument"], "device": full["device"],
        "mmap_forced": full["mmap_forced"],
        "baseline": base, "full": full, "chunked": chk, "chunked_ckpt": ckpt,
        "attributable_mb": {"full": attr(full), "chunked": attr(chk),
                            "chunked_ckpt": attr(ckpt)},
        "ratio_full_over_chunked": round(attr(full) / attr(chk), 2),
        "ratio_full_over_ckpt": round(attr(full) / attr(ckpt), 2),
        "losses_identical": len(set(round(x["loss"], 9)
                                    for x in (full, chk, ckpt))) == 1,
        "max_spread_mb": max(x["spread_mb"] for x in (base, full, chk, ckpt)),
        "all_rows_deterministic": all(x["spread_mb"] < 2.0
                                      for x in (base, full, chk, ckpt)),
        # exact, machine-independent
        "saved_bytes_mb": {"full": s_full["saved_mb"],
                           "chunked": s_chk["saved_mb"],
                           "chunked_ckpt": s_ckpt["saved_mb"]},
        "chunking_saves_nothing_in_the_graph":
            s_chk["saved_mb"] == s_full["saved_mb"],
        "ckpt_graph_reduction": round(s_full["saved_mb"] / s_ckpt["saved_mb"], 2),
        "saved_bytes_deterministic": all(
            x["deterministic"] for x in (s_full, s_chk, s_ckpt)),
        "analytic_logit_mb_full": round(analytic_full, 1),
        "analytic_logit_mb_chunk": round(analytic_chunk, 1),
        "analytic_ratio": round(analytic_full / analytic_chunk, 1),
        "slowdown_ckpt_vs_full": round(ckpt["median_sec"] / full["median_sec"], 2),
        "speedup_chunked_vs_full": round(full["median_sec"] / chk["median_sec"], 2),
        "chunked_is_faster_than_full": chk["median_sec"] < full["median_sec"],
    }
    return R


# ==========================================================================
# 4. document packing + boundary masking (needs a trained model)
# ==========================================================================
def stage_boundary(R, ckpt_path="mtp_ckpt.pt"):
    """
    Item 4, with the control the single-pair version was missing.

    A cross-document seam should be expensive BECAUSE the documents are
    unrelated -- not because seams are expensive. So we also pack two passages
    of the SAME document, where the "seam" is a fiction: nothing actually
    changes there. If that control costs as much, the effect is not about
    document boundaries at all.
    """
    torch.manual_seed(0)
    cfg = Config(block_size=T, n_heads_out=2)
    m = NanoLM(cfg)
    p = os.path.join(HERE, ckpt_path)
    trained = os.path.exists(p)
    if trained:
        m.load_state_dict(torch.load(p, map_location="cpu"))
    else:
        print("  WARNING: no mtp_ckpt.pt -- boundary numbers come from an "
              "UNTRAINED model.\n  Run: python3 train_mtp.py --total 700 --chunk 60")
    m.eval()

    text = open(corpus_path()).read()
    shake_a = ENC.encode(text[2000:2600])[:64]
    shake_b = ENC.encode(text[40000:40600])[:64]
    shake_c = ENC.encode(text[70000:70600])[:64]
    code_a = ENC.encode("def compute_loss(logits, targets):\n"
                        "    return F.cross_entropy(logits.reshape(-1, V), "
                        "targets.reshape(-1))\nimport torch\n")[:64]
    json_a = ENC.encode('{"vocab_size": 50257, "n_layer": 4, "tie": true, '
                        '"lr": 0.0003, "steps": 700}')[:64]

    pairs = [
        ("shakespeare -> python", shake_a, code_a, False),
        ("python -> shakespeare", code_a, shake_b, False),
        ("shakespeare -> json", shake_c, json_a, False),
        ("shakespeare -> shakespeare (CONTROL)", shake_a, shake_b, True),
        ("python -> json", code_a, json_a, False),
    ]

    rows = []
    for name, da, db, is_control in pairs:
        a_, b_ = len(da), len(db)
        packed = torch.tensor(da + db)[None, :]
        doc_ids = build_doc_ids([a_, b_])
        inp, tgt = shifted_targets(packed)
        lmask = boundary_loss_mask(doc_ids, offset=1)[None, :]
        assert int((~lmask).sum()) == 1

        with torch.no_grad():
            h = m(inp)
            logits = m.head(h)
            per_tok, _ = masked_cross_entropy(logits, tgt, reduction="none")
            loss_none, n_none = masked_cross_entropy(logits, tgt)
            loss_lm, n_lm = masked_cross_entropy(logits, tgt, mask=lmask)
            amask = document_attn_mask(doc_ids)[:, :, :-1, :-1]
            logits_dc = m.head(m(inp, attn_mask=amask))
            loss_both, _ = masked_cross_entropy(logits_dc, tgt, mask=lmask)
            per_dc, _ = masked_cross_entropy(logits_dc, tgt, reduction="none")

        bi = a_ - 1
        rows.append({
            "pair": name, "is_control": is_control,
            "len_a": a_, "len_b": b_, "slots": n_none,
            "boundary_slot_index": bi,
            "boundary_input_str": ENC.decode([int(inp[0, bi])]),
            "boundary_target_str": ENC.decode([int(tgt[0, bi])]),
            "boundary_token_loss": round(float(per_tok[bi]), 4),
            "mean_loss_all_slots": round(float(loss_none), 4),
            "ratio_boundary_to_mean": round(float(per_tok[bi]) / float(loss_none), 2),
            "worse_than_uniform": float(per_tok[bi]) > math.log(cfg.vocab_size),
            "loss_no_mask": round(float(loss_none), 4),
            "loss_lossmask_only": round(float(loss_lm), 4),
            "loss_lossmask_plus_docattn": round(float(loss_both), 4),
            "delta_from_loss_mask": round(float(loss_none) - float(loss_lm), 4),
            "delta_from_doc_attention": round(float(loss_lm) - float(loss_both), 4),
            "doc_b_mean_causal": round(float(per_tok[a_:].mean()), 4),
            "doc_b_mean_docattn": round(float(per_dc[a_:].mean()), 4),
        })

    real = [r for r in rows if not r["is_control"]]
    ctrl = [r for r in rows if r["is_control"]]
    head = rows[0]

    R["boundary"] = {
        "trained_checkpoint_used": trained,
        "n_pairs": len(rows),
        "pairs": rows,
        # headline pair, kept so the README's single-example story stays sourced
        "len_doc_a": head["len_a"], "len_doc_b": head["len_b"],
        "packed_len": head["len_a"] + head["len_b"],
        "boundary_slot_index": head["boundary_slot_index"],
        "boundary_input_str": head["boundary_input_str"],
        "boundary_target_str": head["boundary_target_str"],
        "boundary_token_loss": head["boundary_token_loss"],
        "mean_loss_all_slots": head["mean_loss_all_slots"],
        "n_slots_unmasked": head["slots"],
        "loss_no_mask": head["loss_no_mask"],
        "loss_lossmask_only": head["loss_lossmask_only"],
        "loss_lossmask_plus_docattn": head["loss_lossmask_plus_docattn"],
        "delta_from_loss_mask": head["delta_from_loss_mask"],
        "delta_from_doc_attention": head["delta_from_doc_attention"],
        "boundary_loss_vs_mean_ratio": head["ratio_boundary_to_mean"],
        # aggregate
        "cross_domain_ratio_mean": round(
            sum(r["ratio_boundary_to_mean"] for r in real) / len(real), 2),
        "cross_domain_ratio_min": min(r["ratio_boundary_to_mean"] for r in real),
        "control_ratio": ctrl[0]["ratio_boundary_to_mean"],
        "control_boundary_loss": ctrl[0]["boundary_token_loss"],
        "control_cheaper_than_all_cross_domain": all(
            ctrl[0]["ratio_boundary_to_mean"] < r["ratio_boundary_to_mean"] for r in real),
        "all_cross_domain_above_mean": all(r["ratio_boundary_to_mean"] > 1 for r in real),
        "n_cross_domain_worse_than_uniform": sum(r["worse_than_uniform"] for r in real),
        "doc_attention_helped_in_n_pairs": sum(
            r["delta_from_doc_attention"] > 0 for r in rows),
        "doc_attention_effect_is_inconsistent": (
            0 < sum(r["delta_from_doc_attention"] > 0 for r in rows) < len(rows)),
        "max_abs_doc_attention_delta": round(
            max(abs(r["delta_from_doc_attention"]) for r in rows), 4),

        # --- what the control actually showed -------------------------
        # ratio-to-mean is a bad metric: the mean itself swings from 4.04 to
        # 11.55 across pairs, so the control scores HIGHEST on ratio while
        # being cheap in absolute terms. Absolute loss vs ln(V) is the metric
        # that means the same thing in every pair.
        "ln_V": round(math.log(cfg.vocab_size), 4),
        "mean_loss_range": [min(r["mean_loss_all_slots"] for r in rows),
                            max(r["mean_loss_all_slots"] for r in rows)],
        "control_ratio_is_highest": all(
            ctrl[0]["ratio_boundary_to_mean"] >= r["ratio_boundary_to_mean"] for r in real),
        "control_abs_below_uniform": not ctrl[0]["worse_than_uniform"],
        "ratio_metric_is_misleading": (
            all(ctrl[0]["ratio_boundary_to_mean"] >= r["ratio_boundary_to_mean"] for r in real)
            and not ctrl[0]["worse_than_uniform"]),

        # --- direction, not the seam ----------------------------------
        # leaving the model's dominant domain (Shakespeare) is expensive;
        # entering it is cheap. That is a fact about the model's prior.
        "leaving_shakespeare": [r["boundary_token_loss"] for r in rows
                                if r["pair"].startswith("shakespeare ->")
                                and not r["is_control"]],
        "entering_shakespeare": [r["boundary_token_loss"] for r in rows
                                 if r["pair"].endswith("-> shakespeare")],
        "cost_is_directional": (
            min(r["boundary_token_loss"] for r in rows
                if r["pair"].startswith("shakespeare ->") and not r["is_control"])
            > max(r["boundary_token_loss"] for r in rows
                  if r["pair"].endswith("-> shakespeare"))),
    }
    return R


# ==========================================================================
# Part 2: second head predicting t+2
# ==========================================================================
def _slope(sub):
    """OLS slope of gap vs step, per 100 steps, with standard error."""
    n = len(sub)
    xs = [x["step"] for x in sub]; ys = [x["gap"] for x in sub]
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx; a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = (sum(r * r for r in resid) / (n - 2)) ** 0.5
    se = sd / (sxx ** 0.5)
    return {"n": n, "slope_per_100": round(b * 100, 4),
            "se_per_100": round(se * 100, 4),
            "t": round(b / se, 2), "resid_sd": round(sd, 4),
            "ci95_per_100": [round((b - 2 * se) * 100, 4),
                             round((b + 2 * se) * 100, 4)]}


def stage_mtp(R):
    """Folds the resumable run in train_mtp.py into results.json."""
    hist = json.load(open(os.path.join(HERE, "mtp_hist.json")))
    first, last = hist[0], hist[-1]
    gaps = [x["gap"] for x in hist]
    # first logged step where head 1 is genuinely worse than head 0
    crossover = next((x["step"] for x in hist if x["gap"] > 0), None)
    R["mtp"] = {
        "steps": last["step"], "lr": 3e-4, "B": B, "T": T,
        "head0_predicts": "t+1", "head1_predicts": "t+2",
        "trained_jointly_on": "L1 + L2",
        "history": hist,
        "l1_first": first["l1"], "l2_first": first["l2"],
        "l1_final": last["l1"], "l2_final": last["l2"],
        "sum_final": last["sum"],
        "gap_first": first["gap"], "gap_final": last["gap"],
        "gap_max": max(gaps), "gap_min": min(gaps),
        "gap_crossover_step": crossover,
        "ppl1_final": round(math.exp(last["l1"]), 1),
        "ppl2_final": round(math.exp(last["l2"]), 1),
        "gap_starts_negative": first["gap"] < 0,
        "gap_ends_positive": last["gap"] > 0,
        "l2_above_l1_after_crossover": all(
            x["gap"] > 0 for x in hist if x["step"] >= crossover),
        # Is the gap still growing? Fit the slope in windows. A single flat
        # stretch is not a plateau -- at step 300 the gap LOOKED flat over the
        # last 4 points while its true slope was still +0.075/100 steps.
        "gap_slope_windows": {
            f"{lo}-{hi}": _slope([x for x in hist if lo <= x["step"] <= hi])
            for lo, hi in [(40, 300), (300, 500), (500, 700), (40, last["step"])]
            if len([x for x in hist if lo <= x["step"] <= hi]) >= 4
            and hi <= last["step"]
        },
        "l1_still_falling": hist[-10]["l1"] > last["l1"] or last["l1"] > min(
            x["l1"] for x in hist),
        "converged": False,
    }
    w = R["mtp"]["gap_slope_windows"]
    early, late = w.get("40-300"), w.get("500-700")
    if early and late:
        R["mtp"]["growth_decelerates"] = late["slope_per_100"] < early["slope_per_100"] / 3
        R["mtp"]["early_slope_significant"] = early["t"] > 3
        R["mtp"]["late_slope_indistinguishable_from_zero"] = abs(late["t"]) < 2
        # a flat-looking window is not a floor: the CI still admits growth
        R["mtp"]["late_ci_admits_continued_growth"] = late["ci95_per_100"][1] > 0.02
    return R


# ==========================================================================
# seed sweep: is Part 2 a property of the setup, or of seed 1337?
# ==========================================================================
def stage_seeds(R, seeds=(1337, 7, 2024), horizon=300):
    """
    The Part 2 narrative rests on three claims. Each is checked across seeds.
      1. the step-1 gap sign is NOISE
      2. head 1 crosses above head 0 and stays there
      3. the gap magnitude at the horizon is a property of the task
    """
    runs = {}
    for sd in seeds:
        p = (os.path.join(HERE, "mtp_hist.json") if sd == 1337
             else os.path.join(HERE, f"seed{sd}_hist.json"))
        if not os.path.exists(p):
            continue
        h = [x for x in json.load(open(p)) if x["step"] <= horizon]
        cross = next((x["step"] for x in h if x["gap"] > 0), None)
        after = [x for x in h if cross is not None and x["step"] > cross]
        runs[str(sd)] = {
            "gap_at_step1": h[0]["gap"],
            "gap_at_horizon": h[-1]["gap"],
            "crossover_step": cross,
            "slope_40_to_horizon": _slope(
                [x for x in h if 40 <= x["step"] <= horizon])["slope_per_100"],
            "stays_positive_after_crossover": all(x["gap"] > 0 for x in after),
            "l1_final": h[-1]["l1"], "l2_final": h[-1]["l2"],
        }

    g1 = [r["gap_at_step1"] for r in runs.values()]
    gh = [r["gap_at_horizon"] for r in runs.values()]
    cr = [r["crossover_step"] for r in runs.values()]
    sl = [r["slope_40_to_horizon"] for r in runs.values()]

    R["seeds"] = {
        "seeds": list(runs), "horizon": horizon, "runs": runs,
        "step1_gap_range": [min(g1), max(g1)],
        "step1_sign_flips_across_seeds": min(g1) < 0 < max(g1),
        "horizon_gap_range": [min(gh), max(gh)],
        "horizon_gap_spread": round(max(gh) - min(gh), 4),
        "horizon_gap_mean": round(sum(gh) / len(gh), 4),
        "horizon_gap_relative_spread": round((max(gh) - min(gh)) / (sum(gh) / len(gh)), 3),
        "crossover_range": [min(cr), max(cr)],
        "slope_range": [min(sl), max(sl)],
        "all_seeds_gap_positive_at_horizon": all(x > 0 for x in gh),
        "all_seeds_stay_positive_after_crossover": all(
            r["stays_positive_after_crossover"] for r in runs.values()),
        "all_seeds_slope_positive": all(x > 0 for x in sl),
        # the headline: magnitude agrees within ~10% across seeds
        "slope_mean": round(sum(sl) / len(sl), 4),
        "slope_relative_spread": round((max(sl) - min(sl)) / (sum(sl) / len(sl)), 3),
        "slope_more_stable_than_horizon_gap": (
            (max(sl) - min(sl)) / (sum(sl) / len(sl))
            < (max(gh) - min(gh)) / (sum(gh) / len(gh))),
        "horizon_gap_agrees_within_15pct": (
            (max(gh) - min(gh)) / (sum(gh) / len(gh)) < 0.15),
    }
    return R


# ==========================================================================
# the warning: a wrong shift produces a beautiful loss curve
# ==========================================================================
def stage_shiftlong(R, steps=400, lr=3e-4, chunk=None):
    """
    The README previously asserted that the reversed shift "would get there"
    with more training. That was a prediction stated as a fact. This tests it:
    train the reversed convention 6.7x longer and see whether the training loss
    actually collapses toward zero.
    """
    import pickle
    ST = os.path.join(HERE, "shiftlong_state.pt")
    torch.manual_seed(1337)
    cfg = Config(block_size=T, tie_weights=True)
    m = NanoLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    step0, hist = 0, []
    if os.path.exists(ST):
        st = torch.load(ST, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step0, hist = st["step"], st["hist"]
    gen = batches(corpus_tokens(), B, T, seed=11)
    for _ in range(step0):
        next(gen)
    end = min(step0 + chunk, steps) if chunk else steps
    for step in range(step0 + 1, end + 1):
        seq = next(gen)[:, :T + 1]
        inp, tgt = seq[:, 1:T + 1], seq[:, :T]      # predict the PREVIOUS token
        loss, _ = masked_cross_entropy(m.head(m(inp)), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append({"step": step, "loss": round(float(loss.detach()), 4)})
            print(f"  reversed step {step:4d}  loss {float(loss.detach()):.4f}", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                "step": end, "hist": hist}, ST)
    if end < steps:
        print(f"[shiftlong] {end}/{steps} MORE")
        R["shiftlong"] = {"incomplete": True, "step": end, "history": hist}
        return R
    print(f"[shiftlong] {end}/{steps} DONE")
    R["shiftlong"] = {
        "steps": steps, "history": hist,
        "loss_final": hist[-1]["loss"],
        "loss_at_50": next((h["loss"] for h in hist if h["step"] == 50), None),
        "collapsed_below_1_nat": hist[-1]["loss"] < 1.0,
        "collapsed_below_half_of_start": hist[-1]["loss"] < hist[0]["loss"] / 2,
    }
    return R


def stage_shiftbug(R, steps=60, lr=3e-4, only=None):
    def train(mode):
        torch.manual_seed(1337)
        cfg = Config(block_size=T, tie_weights=True)
        m = NanoLM(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        data = corpus_tokens()
        gen = batches(data, B, T, seed=11)
        hist = []
        for step in range(1, steps + 1):
            seq = next(gen)[:, :T + 1]
            if mode == "correct":
                inp, tgt = seq[:, :T], seq[:, 1:T + 1]
            elif mode == "no_shift":
                inp, tgt = seq[:, :T], seq[:, :T]
            elif mode == "reversed":
                inp, tgt = seq[:, 1:T + 1], seq[:, :T]
            h = m(inp)
            loss, _ = masked_cross_entropy(m.head(h), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step % 25 == 0 or step == 1:
                hist.append({"step": step, "loss": round(float(loss), 4)})
        # held-out TRUE next-token loss — the only thing that matters
        val = next(batches(data, B, T, seed=999))[:, :T + 1]
        with torch.no_grad():
            h = m(val[:, :T])
            true_loss, _ = masked_cross_entropy(m.head(h), val[:, 1:T + 1])
        return hist, round(float(true_loss), 4)

    out = R.get("shiftbug", {})
    modes = [only] if only else ["correct", "no_shift", "reversed"]
    for mode in modes:
        print(f"  shiftbug: {mode}", flush=True)
        hist, true_loss = train(mode)
        out[mode] = {
            "history": hist,
            "train_loss_final": hist[-1]["loss"],
            "held_out_true_next_token_loss": true_loss,
            "held_out_true_ppl": round(math.exp(true_loss), 1),
        }
    if not all(k in out for k in ("correct", "no_shift", "reversed")):
        R["shiftbug"] = out
        return R
    out["steps"] = steps
    out["verdict"] = {
        "no_shift_train_loss_beats_correct":
            out["no_shift"]["train_loss_final"] < out["correct"]["train_loss_final"],
        "no_shift_true_loss_worse_than_correct":
            out["no_shift"]["held_out_true_next_token_loss"]
            > out["correct"]["held_out_true_next_token_loss"],
    }
    R["shiftbug"] = out
    return R


# ==========================================================================

STAGES = {
    "shapes": stage_shapes, "shift": stage_shift, "pad": stage_pad,
    "ppl": stage_ppl, "params": stage_params, "memory": stage_memory,
    "mtp": stage_mtp, "boundary": stage_boundary, "shiftbug": stage_shiftbug,
    "shiftlong": stage_shiftlong, "seeds": stage_seeds,
}
ORDER = ["shapes", "shift", "pad", "ppl", "params", "memory",
         "mtp", "boundary", "shiftbug", "shiftlong", "seeds"]

if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = ORDER
    R = load()
    R.setdefault("_meta", {})
    R["_meta"].update({
        "torch": torch.__version__, "device": DEVICE,
        "tokenizer": "gpt2/tiktoken", "vocab_size": ENC.n_vocab,
    })
    for name in want:
        print(f"[{name}]", flush=True)
        t = time.time()
        R = STAGES[name](R)
        print(f"[{name}] done in {time.time()-t:.1f}s", flush=True)
        save(R)
    print("wrote", RESULTS)
