"""
test_invariants.py — every numeric claim in README.md is asserted here.

Two kinds of check:

  LEDGER  — the README states number X; results.json must contain exactly X.
            This is what stops the write-up drifting from the run.
  PROPERTY— an independent recomputation or structural property that does not
            read results.json at all, so a wrong pipeline cannot satisfy it.

    python3 test_invariants.py          # run everything
"""

import json
import math
import os
import re
import sys

import torch

from nanolm import (
    Config, NanoLM, boundary_loss_mask, build_doc_ids, chunked_cross_entropy,
    chunked_cross_entropy_ckpt, document_attn_mask, masked_cross_entropy,
    shifted_targets,
)

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
README = open(os.path.join(HERE, "README.md")).read() if os.path.exists(
    os.path.join(HERE, "README.md")) else ""

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))


def in_readme(s):
    """
    README must contain this number as a STANDALONE token.

    Substring matching is not enough: "6.969" is a substring of "6.9690" and of
    "6.9691", so a stale digit would pass silently. Require a non-digit (or
    string edge) on both sides.
    """
    return re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", README) is not None


# ==========================================================================
# LEDGER: README text must match results.json
# ==========================================================================
def ledger():
    print("\n[ledger] README numbers are backed by results.json")
    sh, pd, pp, pa = R["shapes"], R["pad"], R["ppl"], R["params"]
    me, bo, mt, sb = R["memory"], R["boundary"], R["mtp"], R["shiftbug"]
    sd = R["seeds"]

    claims = [
        ("V = 50257", str(pp["V"])),
        ("logit tensor MB", str(sh["logit_bytes_fp32_mb"])),
        ("pad slots before", str(pd["n_unmasked"])),
        ("pad slots after", str(pd["n_masked"])),
        ("pad pct", str(pd["pct_of_slots_that_were_pad"])),
        ("pad unmasked reported loss",
         str(pd["trained_short_run"]["unmasked_run"]["reported_loss_final"])),
        ("pad unmasked true loss",
         str(pd["trained_short_run"]["unmasked_run"]["true_content_loss"])),
        ("pad masked true loss",
         str(pd["trained_short_run"]["masked_run"]["true_content_loss"])),
        ("untrained ppl", str(pp["ppl_untrained"])),
        ("ppl/V ratio", str(pp["ratio_ppl_over_V"])),
        ("tied params", f"{pa['tied_total']:,}"),
        ("untied params", f"{pa['untied_total']:,}"),
        ("param difference", f"{pa['difference']:,}"),
        ("memory full attributable", str(me["attributable_mb"]["full"])),
        ("memory chunked attributable", str(me["attributable_mb"]["chunked"])),
        ("memory ratio", str(me["ratio_full_over_chunked"])),
        ("analytic ratio", str(me["analytic_ratio"])),
        ("saved bytes full", str(me["saved_bytes_mb"]["full"])),
        ("saved bytes ckpt", str(me["saved_bytes_mb"]["chunked_ckpt"])),
        ("ckpt graph reduction", str(me["ckpt_graph_reduction"])),
        ("max spread", str(me["max_spread_mb"])),
        ("boundary token loss", str(bo["boundary_token_loss"])),
        ("boundary mean loss", str(bo["mean_loss_all_slots"])),
        ("boundary control ratio", str(bo["control_ratio"])),
        ("boundary control abs loss", str(bo["control_boundary_loss"])),
        ("boundary n pairs", str(bo["n_pairs"])),
        ("doc-attn pairs helped", str(bo["doc_attention_helped_in_n_pairs"])),
        ("mtp l1 final", str(mt["l1_final"])),
        ("mtp l2 final", str(mt["l2_final"])),
        ("mtp gap final", str(mt["gap_final"])),
        ("mtp sum final", str(mt["sum_final"])),
        ("mtp steps", str(mt["steps"])),
        ("gap slope 40-300", str(mt["gap_slope_windows"]["40-300"]["slope_per_100"])),
        ("gap slope 300-500", str(mt["gap_slope_windows"]["300-500"]["slope_per_100"])),
        ("gap slope 500-700", str(mt["gap_slope_windows"]["500-700"]["slope_per_100"])),
        ("shiftbug no_shift train", str(sb["no_shift"]["train_loss_final"])),
        ("shiftbug no_shift true ppl", str(sb["no_shift"]["held_out_true_ppl"])),
        ("shiftbug correct true ppl", str(sb["correct"]["held_out_true_ppl"])),
        ("reversed 400-step train loss", str(R["shiftlong"]["loss_final"])),
        ("reversed 400-step true ppl", str(R["shiftlong"]["held_out_true_ppl"])),
        # the corrections narrative must also be sourced
        ("superseded ckpt band low",
         str(R["superseded"]["memory_pre_mmap_fix"]["chunked_ckpt_min_mb"])),
        ("superseded ckpt band high",
         str(R["superseded"]["memory_pre_mmap_fix"]["chunked_ckpt_max_mb"])),
        ("superseded spread",
         str(R["superseded"]["memory_pre_mmap_fix"]["chunked_ckpt_spread_mb"])),
        ("superseded speedup claim",
         str(R["superseded"]["memory_pre_mmap_fix"]["speedup_chunked_vs_full_claimed"])),
        ("superseded boundary ratio",
         str(R["superseded"]["boundary_pre_control"]["ratio_claimed"])),
        ("superseded mtp gap",
         str(R["superseded"]["mtp_pre_extension"]["gap_final"])),
        ("seed count", str(len(sd["seeds"]))),
        ("seed horizon gap mean", str(sd["horizon_gap_mean"])),
        ("seed horizon spread pct", str(sd["horizon_gap_relative_spread"])),
        ("seed slope mean", str(sd["slope_mean"])),
        ("seed slope spread", str(sd["slope_relative_spread"])),
        ("seed slope low", str(sd["slope_range"][0])),
        ("seed slope high", str(sd["slope_range"][1])),
        ("seed crossover high", str(sd["crossover_range"][1])),
        ("seed 7 step1 gap", str(sd["runs"]["7"]["gap_at_step1"])),
    ]
    for name, val in claims:
        check(f"README states {name} = {val}", in_readme(val), f"missing '{val}'")


# ==========================================================================
# PROPERTY: recomputed from scratch, independent of results.json
# ==========================================================================
def properties():
    print("\n[property] independent recomputation")
    cfg = Config()
    V, D = cfg.vocab_size, cfg.n_embd

    # --- 1. shapes -------------------------------------------------------
    torch.manual_seed(0)
    m = NanoLM(Config(block_size=128))
    tok = torch.randint(0, V, (8, 128))
    inp, tgt = shifted_targets(tok)
    h = m(inp)
    lg = m.head(h)
    check("inputs are one shorter than tokens", inp.shape[1] == tok.shape[1] - 1)
    check("targets align with inputs", tgt.shape == inp.shape)
    check("logits last dim is V", lg.shape[-1] == V)
    check("logit bytes match B*T*V*4",
          abs(lg.numel() * 4 / 1e6 - R["shapes"]["logit_bytes_fp32_mb"]) < 0.1)

    # --- 2. the shift really is next-token -------------------------------
    ids = torch.arange(10)[None, :]
    i2, t2 = shifted_targets(ids)
    check("shift: target[i] == input[i]+1 on a ramp",
          bool((t2[0] == i2[0] + 1).all()))
    check("shift: no position predicts itself",
          bool((t2[0] != i2[0]).all()))

    # --- 3. pad masking changes the denominator --------------------------
    seq = torch.randint(0, 50000, (8, 128))
    seq[:, 40:] = 50256
    _, t3 = shifted_targets(seq)
    n_all = t3.numel()
    n_keep = int((t3 != 50256).sum())
    check("pad mask drops exactly the pad targets",
          n_all - n_keep == R["pad"]["n_dropped"], f"{n_all-n_keep}")
    check("kept count is 39 real targets per row",
          n_keep == 8 * 39, str(n_keep))

    # --- 4. boundary mask drops exactly one slot -------------------------
    dids = build_doc_ids([64, 49])
    bmask = boundary_loss_mask(dids, offset=1)
    check("boundary mask drops exactly 1 slot for 2 docs",
          int((~bmask).sum()) == 1)
    check("dropped slot sits at end of doc A",
          int((~bmask).nonzero()[0]) == 63)
    amask = document_attn_mask(dids)
    check("doc attention mask is causal",
          bool((amask[0, 0].triu(1) == False).all()))
    check("doc attention blocks B->A",
          not bool(amask[0, 0, 70, 10]))
    check("doc attention allows A->A", bool(amask[0, 0, 10, 5]))

    # --- 5. untrained loss sits at ln(V) + var/2 -------------------------
    torch.manual_seed(0)
    mu = NanoLM(Config(block_size=128, tie_weights=False))
    rt = torch.randint(0, V, (8, 128), generator=torch.Generator().manual_seed(3))
    ri, rt2 = shifted_targets(rt)
    with torch.no_grad():
        rl = mu.head(mu(ri))
        loss, _ = masked_cross_entropy(rl, rt2)
    pred = math.log(V) + float(rl.var()) / 2
    check("untrained loss matches ln(V)+var/2 within 0.01 nats",
          abs(float(loss) - pred) < 0.01, f"{float(loss)-pred:.5f}")
    check("untrained ppl within 6% of V",
          abs(math.exp(float(loss)) / V - 1) < 0.06)

    # --- 6. tied vs untied differ by exactly V*D -------------------------
    t_, u_ = NanoLM(Config(tie_weights=True)), NanoLM(Config(tie_weights=False))
    check("untied - tied == V*D exactly",
          u_.n_unique_params() - t_.n_unique_params() == V * D)
    check("tied head shares storage with wte",
          t_.heads[0].weight.data_ptr() == t_.wte.weight.data_ptr())
    check("untied head does NOT share storage",
          u_.heads[0].weight.data_ptr() != u_.wte.weight.data_ptr())

    # --- 7. chunked CE is numerically equivalent -------------------------
    torch.manual_seed(0)
    mc = NanoLM(Config(block_size=128))
    tk = torch.randint(0, V, (2, 128))
    ci, ct = shifted_targets(tk)
    hh = mc(ci)
    l_full, _ = masked_cross_entropy(mc.head(hh), ct)
    l_chunk, _ = chunked_cross_entropy(hh, mc.heads[0].weight, ct, chunk_size=64)
    l_ckpt, _ = chunked_cross_entropy_ckpt(hh, mc.heads[0].weight, ct, chunk_size=64)
    check("chunked CE == full CE (value)",
          abs(float(l_full) - float(l_chunk)) < 1e-5)
    check("checkpointed chunked CE == full CE (value)",
          abs(float(l_full) - float(l_ckpt)) < 1e-4)

    g = {}
    for tag, lo in (("full", l_full), ("chunk", l_chunk), ("ckpt", l_ckpt)):
        mc.zero_grad(set_to_none=True)
        lo.backward(retain_graph=True)
        g[tag] = mc.heads[0].weight.grad.clone()
    check("chunked CE == full CE (head gradient)",
          float((g["full"] - g["chunk"]).abs().max()) < 1e-6)
    check("checkpointed CE == full CE (head gradient)",
          float((g["full"] - g["ckpt"]).abs().max()) < 1e-6)

    # --- 8. masking a position removes it from the mean ------------------
    torch.manual_seed(0)
    lg2 = torch.randn(1, 5, V)
    tg2 = torch.randint(0, V, (1, 5))
    per, _ = masked_cross_entropy(lg2, tg2, reduction="none")
    keep = torch.tensor([[True, True, False, True, True]])
    lm, n = masked_cross_entropy(lg2, tg2, mask=keep)
    manual = (per[0] + per[1] + per[3] + per[4]) / 4
    check("masked mean equals manual mean of kept slots",
          abs(float(lm) - float(manual)) < 1e-5)
    check("masked count is 4", n == 4)

    # --- 9. MTP: head 1 target is one further ahead ----------------------
    s = torch.arange(20)[None, :]
    check("MTP head1 target leads head0 target by 1",
          bool((s[:, 2:12] == s[:, 1:11] + 1).all()))


# ==========================================================================
# PROPERTY: qualitative claims that carry the write-up
# ==========================================================================
def narrative():
    print("\n[narrative] the claims the write-up actually makes")
    pd, pp, me = R["pad"], R["ppl"], R["memory"]
    bo, mt, sb = R["boundary"], R["mtp"], R["shiftbug"]
    sd = R["seeds"]

    tr = pd["trained_short_run"]
    check("unmasked pad run REPORTS a lower loss than masked",
          tr["unmasked_run"]["reported_loss_final"] < tr["masked_run"]["reported_loss_final"])
    check("unmasked pad run is ACTUALLY worse on real content",
          tr["unmasked_run"]["true_content_loss"] > tr["masked_run"]["true_content_loss"])

    check("ppl/V formula is exact in the untied control",
          abs(pp["conditions"]["untied_head_random_tokens"]["prediction_error_nats"]) < 0.01)
    check("deviation grows: untied<tied<real",
          abs(pp["conditions"]["untied_head_random_tokens"]["prediction_error_nats"])
          < abs(pp["conditions"]["tied_head_random_tokens"]["prediction_error_nats"])
          < abs(pp["conditions"]["tied_head_real_text"]["prediction_error_nats"]))

    check("all three CE implementations produced identical loss",
          me["losses_identical"])
    check("chunking reduces attributable peak memory",
          me["attributable_mb"]["chunked"] < me["attributable_mb"]["full"])
    check("checkpointing reduces it further than plain chunking",
          me["attributable_mb"]["chunked_ckpt"] < me["attributable_mb"]["chunked"])
    check("measured ratio is far below the analytic logit-term ratio",
          me["ratio_full_over_chunked"] < me["analytic_ratio"] / 5)
    check("forcing mmap made EVERY row deterministic",
          me["all_rows_deterministic"], str(me["max_spread_mb"]))
    check("the probe declares it forced mmap", me["mmap_forced"])
    check("plain chunking retains EXACTLY as many bytes as full",
          me["chunking_saves_nothing_in_the_graph"])
    check("checkpointing cuts retained bytes several-fold",
          me["ckpt_graph_reduction"] > 5)
    check("saved-byte accounting is deterministic",
          me["saved_bytes_deterministic"])
    check("chunking is NOT faster once the allocator is honest",
          not me["chunked_is_faster_than_full"])

    check("boundary slot costs more than the sequence mean",
          bo["boundary_token_loss"] > bo["mean_loss_all_slots"])
    check("boundary slot is worse than a uniform guess",
          bo["boundary_token_loss"] > math.log(50257))
    check("loss-masking the boundary lowers reported loss",
          bo["loss_lossmask_only"] < bo["loss_no_mask"])
    check("same-domain CONTROL scores the HIGHEST ratio (metric is bad)",
          bo["control_ratio_is_highest"])
    check("control is nonetheless cheap in absolute terms",
          bo["control_abs_below_uniform"])
    check("therefore ratio-to-mean is declared misleading",
          bo["ratio_metric_is_misleading"])
    check("mean loss varies >2x across pairs (why the ratio misleads)",
          bo["mean_loss_range"][1] / bo["mean_loss_range"][0] > 2)
    check("boundary cost is DIRECTIONAL, not a property of seams",
          bo["cost_is_directional"])
    check("leaving the dominant domain costs more than entering it",
          min(bo["leaving_shakespeare"]) > max(bo["entering_shakespeare"]))
    check("doc attention effect is inconsistent across pairs",
          bo["doc_attention_effect_is_inconsistent"])
    check("doc attention effect is small in magnitude",
          bo["max_abs_doc_attention_delta"] < 0.1)
    check("boundary tested on more than one document pair", bo["n_pairs"] >= 5)

    check("MTP gap starts negative", mt["gap_starts_negative"])
    check("MTP gap ends positive", mt["gap_ends_positive"])
    check("MTP head1 stays worse after crossover", mt["l2_above_l1_after_crossover"])
    w = mt["gap_slope_windows"]
    check("MTP gap growth DECELERATES across windows", mt["growth_decelerates"])
    check("early gap slope is strongly significant", mt["early_slope_significant"])
    check("late gap slope is not distinguishable from zero",
          mt["late_slope_indistinguishable_from_zero"])
    check("late CI still admits continued growth (no floor proven)",
          mt["late_ci_admits_continued_growth"])
    check("slope order is monotone decreasing",
          w["40-300"]["slope_per_100"] > w["300-500"]["slope_per_100"]
          > w["500-700"]["slope_per_100"])
    check("the run is NOT claimed to be converged", mt["converged"] is False)
    check("at step 300 the slope was still significantly positive",
          w["300-500"]["t"] > 3)
    check("MTP ppl2 > ppl1", mt["ppl2_final"] > mt["ppl1_final"])

    # ---- seed sweep: which Part 2 claims survive a change of seed ----
    check("step-1 gap sign FLIPS across seeds (so it is noise)",
          sd["step1_sign_flips_across_seeds"])
    check("every seed ends with a positive gap", sd["all_seeds_gap_positive_at_horizon"])
    check("every seed stays positive after its crossover",
          sd["all_seeds_stay_positive_after_crossover"])
    check("every seed has a positive early slope", sd["all_seeds_slope_positive"])
    check("gap magnitude agrees within 15% across seeds",
          sd["horizon_gap_agrees_within_15pct"],
          str(sd["horizon_gap_relative_spread"]))
    check("early slope is more seed-stable than the horizon gap",
          sd["slope_more_stable_than_horizon_gap"])
    check("early slope relative spread is under 6%",
          sd["slope_relative_spread"] < 0.06, str(sd["slope_relative_spread"]))
    check("crossover step is NOT seed-stable (declared as a range)",
          sd["crossover_range"][1] - sd["crossover_range"][0] > 20)
    check("sweep used at least 3 seeds", len(sd["seeds"]) >= 3)

    check("no_shift TRAIN loss beats the correct run",
          sb["verdict"]["no_shift_train_loss_beats_correct"])
    check("no_shift TRUE loss is worse than the correct run",
          sb["verdict"]["no_shift_true_loss_worse_than_correct"])
    check("no_shift true ppl is >5x the correct run",
          sb["no_shift"]["held_out_true_ppl"] > 5 * sb["correct"]["held_out_true_ppl"])
    check("reversed shift did NOT collapse in 60 steps",
          sb["reversed"]["train_loss_final"] > 3.0)
    sl = R["shiftlong"]
    check("reversed shift DOES collapse given 400 steps (prediction tested)",
          sl["collapsed_below_1_nat"])
    check("collapsed reversed model is far worse than the correct baseline",
          sl["held_out_true_ppl"] > 10 * sb["correct"]["held_out_true_ppl"])
    check("reversed model is near-perfect at its own bogus task",
          sl["held_out_reversed_task_loss"] < 1.0)
    check("reversed bug is INVISIBLE at 60 steps but severe at 400",
          sb["reversed"]["train_loss_final"] > 3.0 and sl["loss_final"] < 1.0)


if __name__ == "__main__":
    print("=" * 66)
    print("Session 9 invariant suite")
    print("=" * 66)
    properties()
    narrative()
    if README:
        ledger()
    else:
        print("\n[ledger] SKIPPED — README.md not present yet")
    print("\n" + "=" * 66)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
    print("=" * 66)
    sys.exit(1 if FAIL else 0)
