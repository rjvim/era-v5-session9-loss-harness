# Session 9 — Loss Functions and Output Heads

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rjvim/era-v5-session9-loss-harness/blob/main/s9_loss_harness.ipynb)

A loss harness for the four lines between a model's output and a scalar, made
observable rather than merely plausible.

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                     tokens[:, 1:].reshape(-1))
```

**Setup.** GPT-2 BPE (V = 50257), a ~16M parameter decoder (D = 256, 4 layers,
4 heads), TinyShakespeare. Runs on a free Colab CPU.

- **[`s9_loss_harness.ipynb`](s9_loss_harness.ipynb)** — the notebook, runs top to bottom
- **[`s9_executed.ipynb`](s9_executed.ipynb)** — the same notebook with outputs, as executed
- **[`results.json`](results.json)** — every number below, machine-readable
- **[`test_invariants.py`](test_invariants.py)** — every number below is checked against `results.json`; structural claims are recomputed from scratch

```bash
pip install -r requirements.txt

# order matters: item 4 evaluates a TRAINED model, so Part 2 runs first.
python3 train_mtp.py --total 700 --chunk 60    # repeat until it prints DONE
python3 seed_sweep.py --seed 7    --total 300 --chunk 60   # replication
python3 seed_sweep.py --seed 2024 --total 300 --chunk 60
python3 run_all.py all                         # every measurement -> results.json
python3 test_invariants.py                     # all assertions, 0 failures
```

`train_mtp.py` checkpoints and exits every `--chunk` steps, so the run survives
short-lived shells. Checkpoints are gitignored, so a fresh clone must train
before `run_all.py boundary` — it warns loudly if you skip that. The corpus
downloads itself on first use.

Every figure here is asserted by `test_invariants.py`. If the run changes and
the README does not, the suite fails.

---

## Part 1 — the seven numbers

| # | Item | Result |
|---|------|--------|
| 1 | Shapes | `logits` = (8, 127, 50257) → **204.2 MB** in fp32 |
| 2 | Shift verified with strings | `target[i] == input[i+1]`, nothing predicts itself |
| 3 | Padding masked | **1016 → 312** contributing slots (**69.3%** were PAD) |
| 4 | Boundary masked | **5** document pairs + a same-domain control |
| 5 | Untrained perplexity | **50891.3** against V = 50257 → ratio **1.0126** |
| 6 | Tied vs untied | **16,048,896** vs **28,914,688**, difference **12,865,792** |
| 7 | Peak memory | full **1566.1 MB**, chunked **510.6 MB** → **3.07×** |

### 1. Shapes

The output head is a classifier with 50,257 classes applied at *every* position.
That one fact drives items 6 and 7 — it is where the parameters are and where
the memory goes. `reshape(-1, V)` is the point at which `B` and `T` stop
existing as separate ideas, so it is worth knowing what went into it.

### 2. The shift, in strings

Printed as decoded strings for the correct convention and both bugs. In the
correct table every `must PREDICT` reappears as the next row's `model READS`; in
the no-shift table every row hands the model the answer. Neither is visible in
integers and neither raises. Backed by a mechanical check on an integer ramp.

### 3. Padding

The count moving from 1016 to 312 is the requirement. It is not the interesting
part — on an **untrained** model, masking barely moves the loss, because every
prediction is equally hopeless. The damage appears only after training:

| run (60 steps, identical seed and data) | REPORTED loss | TRUE content loss |
|---|---|---|
| PAD counted in the loss | **2.2277** | **6.7187** |
| PAD masked out | 6.4615 | **6.3668** |

The unmasked run reports a loss **4.2338 nats lower** while being **0.3519 nats
worse** at the actual job. 69.3% of its slots were free wins on a token carrying
no information.

### 4. Document boundaries — where my first answer was wrong

I originally ran **one** document pair, found the boundary slot cost 1.94× the
sequence mean, and called it a result. Adding four more pairs and a same-domain
control broke that conclusion.

| pair | boundary loss | seq. mean | ratio | > ln(V)? |
|---|---|---|---|---|
| shakespeare → python | **15.8932** | 6.969 | 2.28 | yes |
| python → shakespeare | 8.8233 | 6.716 | 1.31 | no |
| shakespeare → json | 16.9119 | 7.903 | 2.14 | yes |
| **shakespeare → shakespeare (CONTROL)** | 10.2132 | 4.0387 | **2.53** | no |
| python → json | 15.0107 | 11.5495 | 1.30 | yes |

**The control scores the highest ratio of all five.** Its "boundary" is a
fiction — two passages from the same document — so if the metric were measuring
document discontinuity it should have scored lowest. It scored highest because
its *denominator* is small: the sequence mean ranges from **4.0387 to 11.5495**
across pairs. Ratio-to-mean was measuring how well the model knows the
surrounding text, not how surprising the seam is.

Absolute loss against `ln(V) = 10.8249` is the metric that means the same thing
in every pair, and it says something different: **the cost is directional.**
Leaving Shakespeare costs **15.8932–16.9119**; entering it costs **8.8233**.
This model's prior *is* Shakespeare, so transitioning into it is cheap and out
of it is expensive. That is a fact about the model, not about seams.

What survives: the boundary target is still an impossible prediction and still
worth masking — masking it moved the headline pair's loss by **0.085** nats from
a single slot in 112. What does not survive is the claim that a boundary is
intrinsically expensive.

### 5. Untrained perplexity

Measured **50891.3** against V = 50257, ratio **1.0126**. The residual is
arithmetic, not a bug. For logits independent of the target with variance `s²`,
`E[CE] = ln(V) + s²/2`:

| condition | loss | ln(V) + s²/2 | error (nats) |
|---|---|---|---|
| untied head, random tokens | 10.875 | 10.8759 | **−0.000966** |
| tied head, random tokens | 10.8682 | 10.8762 | −0.007994 |
| tied head, real text | 10.8374 | 10.8763 | −0.038807 |

Exact to 1e-3 nats (−0.000966) when its independence assumption holds; tying the head and
then using real text each break that assumption a little more.

### 6. Tied against untied

Difference is **12,865,792 = V × D exactly**, verified by pointer identity. The
embedding is **80.2%** of the tied model and **89.0%** of the untied one.
Untying adds 80% to this model's parameters. At 16M that is a bad trade; at
100B, where `V × D` is a few percent, it is standard. The question is the ratio
of `V × D` to everything else, not a universal rule.

### 7. Peak memory, full vs chunked

At B = 8, T = 256, `logits` is 410.1 MB in fp32 and backward needs a gradient of
the same shape. Cross-entropy is a **sum over positions**, so positions can be
visited in groups — no approximation.

Measured two ways, because one is a proxy and the other is exact.

**Peak RSS** — what the process actually needed. Run with
`MALLOC_MMAP_THRESHOLD_=65536`: glibc otherwise caches freed blocks, so peak RSS
measures the high-water mark of the *allocator*, not of live memory. Without
that flag the checkpoint row drifted by 332.3 MB between identical runs. With it,
**every row is deterministic to 0.2 MB across 7 repeats.**

| variant | median MB | spread | attributable | sec |
|---|---|---|---|---|
| baseline (trunk only) | 715.4 | 0.20 | — | 0.530 |
| full cross-entropy | 2281.5 | 0.20 | **1566.1** | 4.492 |
| chunked | 1226.0 | 0.00 | **510.6** | 4.556 |
| chunked + checkpoint | 1001.0 | 0.10 | **285.6** | 5.281 |

**Ratio: 3.07×** (full / chunked), **5.48×** with checkpointing — both now on a
deterministic instrument, no band.

**Retained bytes** — exactly what the autograd graph saves, counted by
intercepting every saved tensor. Identical on any machine:

| variant | retained MB |
|---|---|
| full | **463.67** |
| chunked | **463.67** |
| chunked + checkpoint | **53.58** |

**Plain chunking retains byte-for-byte the same memory as the full version.**
Not "less than hoped" — *identical*. Autograd saves every chunk's logits, so the
graph is the same size however you slice the forward pass. Checkpointing is what
changes it, by **8.65×**, at **1.18×** wall time.

All three produced a **bit-identical loss** with head gradients agreeing to 1e-8.

---

## Part 2 — a second head predicting `t+2`

Head 0 predicts `t+1`; head 1 predicts `t+2` from the **same hidden state**.
Trained jointly on `L1 + L2` for **700** steps.

| | value |
|---|---|
| L1 final (t+1) | **4.708** (ppl 110.8) |
| L2 final (t+2) | **5.3193** (ppl 204.2) |
| L1 + L2 | **10.0273** |
| gap at step 1 | **−0.039** |
| gap at step 700 | **+0.6114** |
| crossover step | **40** (seed-dependent, see sweep) |

### What happens to the second head's loss, and why

**It starts marginally below head 0, crosses early, and rises for the rest of
the run at a decelerating rate.**

At step 1 the gap is **−0.039** — head 1 is *better*. That is not a result; both
heads are randomly initialised at ln(V) and the sign is noise. Logging step 1 is
what stops "L2 > L1 throughout" being written down as a fact.

**I first ran this to 300 steps and reported that the gap plateaued.** The last
four points were `0.4567, 0.452, 0.4628, 0.453` — flat to the eye. Fitting the
slope in windows shows that was wrong:

| window | slope (nats / 100 steps) | t | 95% CI |
|---|---|---|---|
| 40–300 | **+0.174** ± 0.010 | +16.7 | [0.1532, 0.1948] |
| 300–500 | **+0.0751** ± 0.014 | +5.4 | [0.0473, 0.1029] |
| 500–700 | **+0.0178** ± 0.019 | +0.9 | [−0.0207, 0.0564] |

At step 300, where I called it flat, the true slope was still **+0.0751 per 100
steps at t = 5.4** — unambiguously rising. Four consecutive points inside a
±0.03 noise band are not a plateau. Extending to 700 steps, the gap went on to
**+0.6114**.

What is defensible now: **growth decelerates by roughly 10×** across the run,
and by steps 500–700 the slope is no longer distinguishable from zero. What is
*not* defensible is calling that a floor — the CI still admits continued growth
of up to +0.0564 per 100 steps, and L1 is itself still falling, so the model is
not converged. **`results.json` records `"converged": false`.**

### Does this depend on the seed?

Everything above is one seed. Three seeds, all to step 300:

| seed | gap @ step 1 | crossover | slope/100 (40–300) | gap @ 300 |
|---|---|---|---|---|
| 1337 | −0.039 | 40 | +0.174 | +0.453 |
| 7 | **+0.0893** | 1 | +0.1662 | +0.498 |
| 2024 | −0.0201 | 60 | +0.1749 | +0.5046 |

**Survives.** Every seed opens a positive gap, keeps it positive after crossing,
and has a positive early slope. The slope is the most stable quantity measured
here — **+0.1662 to +0.1749**, mean **+0.1717**, a spread of **0.051** of the
mean. The gap magnitude at step 300 averages **0.4852** with a relative spread
of **0.106** — twice as variable as the slope, but still tight enough that
~0.45–0.50 nats is a property of the task rather than of one run.

**Does not survive: the crossover step.** It ranges from **1** to **60**. Seed 7
happened to start with a *positive* step-1 gap (**+0.0893**), so its "crossover"
is step 1 — which is exactly the noise the step-1 reading is made of. The single
figure of "step 40" quoted above is a seed artifact and should be read as
"early, within the first ~60 steps."

That the step-1 sign flips across seeds (−0.039, +0.0893, −0.0201) is the
cleanest confirmation that it carries no information.

The mechanism is not in doubt even if the asymptote is. Predicting `t+2`
requires integrating over every possible `t+1`, so

```
H(x_{t+2} | x_{<=t})  >=  H(x_{t+1} | x_{<=t})
```

That inequality is a property of the text, not the model, and no training
removes it. What I cannot yet claim is *where* the gap settles — only that it
grows, decelerates, and is bounded below by an information-theoretic floor I
have not measured.

This is also why models are often *trained* with several MTP heads but
*deployed* with one or two: the extra heads shape the trunk early, but
standalone quality degrades with distance fast enough that speculating far ahead
stops paying.

---

## The warning, taken literally

> *A target shift in the incorrect direction can produce a beautiful loss curve.*

Three identical models, identical seed and data, three shift conventions, all
evaluated the same way — true next-token loss on held-out text.

| convention | steps | TRAIN loss | TRUE ppl |
|---|---|---|---|
| correct | 60 | 6.6259 | **535.6** |
| no shift | 60 | **4.4654** | **5750.8** |
| reversed | 60 | 6.6536 | 865.3 |
| **reversed** | **400** | **0.8168** | **21261.2** |

The no-shift run reports a training loss **2.1605 nats better** than correct and
is **10.7× worse** at the only task that matters. It has learned to copy its
input, which is free.

**The reversed run is the more dangerous bug, and I nearly mis-reported it.**
At 60 steps it looks *fine* — 6.6536, indistinguishable from the correct run's
6.6259. My first draft asserted it "would get there" with more training. That
was a prediction stated as a fact, so I tested it: at 400 steps it collapses to
**0.8168**, and reaches **0.812** on held-out data at its own bogus task. It is
near-perfect at predicting the previous token, and its true next-token
perplexity is **21261.2** — **40× worse** than the correct run, and barely
better than uniform guessing.

So the bug is invisible for the first ~100 steps and then produces the most
beautiful loss curve of any run here. A short run does not prove a shift is
correct. **Only printing the strings does.**

---

## Honest negatives and caveats

**1. My first boundary result did not survive a control.** One pair became five
plus a same-domain control; the control scored the *highest* ratio, and the
metric turned out to be measuring the denominator. Corrected above.

**2. My first MTP plateau claim did not survive more training.** Reported at 300
steps, where the gap read **0.453**, from four flat-looking points; the slope
was still significantly positive.
Corrected above. Neither error would have been caught by any test I had written
— both were found by extending the run.

**3. Document-causal attention shows no consistent effect.** Across the five
pairs it helped in **3** and hurt in 2, with a maximum absolute effect of
**0.0833** nats. My single-pair version reported this as a clean negative; it is
better described as noise at this scale. The model was trained *without*
document-causal attention, so retrofitting it at evaluation time is not a test
of the technique. I have not run that experiment and claim nothing about it.

**4. Plain chunking saves nothing at all in the autograd graph.** I originally
inferred this from peak RSS. Counting saved tensors directly proves it exactly:
full and chunked both retain **463.67 MB** — the same number, not a similar one.
Slicing the forward pass does not shrink the graph, because every chunk's logits
are saved for backward. Only `torch.utils.checkpoint` changes it, to **53.58 MB**
(**8.65×**), at **1.18×** wall time.

**5. The analytic and measured memory ratios differ 10×, and both are reported.**
The logit term alone is 410.1 MB vs 12.9 MB = **31.9×**. End-to-end peak RSS is
**3.07×**, because the logit tensor is only part of the footprint. Quoting
either alone would mislead.

**6. The CPU memory instrument was untrustworthy until I fixed the allocator.**
My first version reported `chunked_ckpt` as a band from **1048.1** to **1380.4**
MB — a spread of **332.3** MB — and concluded the measurement needed a GPU. It
did not; it needed `MALLOC_MMAP_THRESHOLD_=65536`. glibc caches freed blocks, so
peak RSS was tracking the allocator's high-water mark rather than live memory.
With large allocations forced through `mmap` (which unmaps on free), the spread
across 7 repeats fell to **0.2** MB and every row became deterministic. Those
pre-fix numbers are preserved in `results.json` under `superseded`, so the
corrections below are sourced rather than remembered.
`mem_probe.py` still switches to `torch.cuda.max_memory_allocated()` on a GPU,
but item 7 no longer depends on having one.

**7. The 700-step deceleration is single-seed.** The seed sweep runs to step 300,
which covers the crossover and the high-signal 40–300 window, but the
decelerating-slope result (0.174 → 0.0751 → 0.0178) rests on seed 1337 alone. I
have not shown the deceleration rate is seed-stable, only that the gap's
existence, sign and magnitude are.

**8. Scale caveat.** Everything here is a 16M-parameter model on TinyShakespeare.
The tied/untied trade, the MTP gap size, the boundary directionality and the
memory ratios are all functions of `V × D` relative to the rest of the model and
of a Shakespeare-dominated prior. None should be extrapolated to a 120B run
without re-measuring.

---

## What I got wrong while building this

Listed because the pattern is more useful than the numbers.

- **Called a plateau at step 300.** Four flat points inside a noise band; the
  slope was +0.0751/100 at t=5.4. Fixed by training 2.3× longer and fitting
  windows instead of eyeballing.
- **Drew a conclusion from n=1 on the boundary.** No control. The control, when
  added, contradicted the framing and exposed the metric as the real problem.
- **Stated an untested prediction as fact** ("it would get there"). It happened
  to be right, which is worse — it would have shipped as an unearned claim.
- **Asserted "L2 > L1 throughout"** before looking at step 1, where the gap is
  −0.039.
- **Reported chunking as 1.52× FASTER than the full version, and explained it
  with a cache-locality story.** It was an allocator artifact. Once large
  allocations go through `mmap`, the speedup is **0.99×** — chunking is neither
  faster nor slower. A tidy mechanism I invented for a number that was noise.
- **Wrote a memory probe without checking reproducibility,** then concluded the
  noise was intrinsic to CPU and needed a GPU to fix. It was a malloc tuning
  flag. Seven repeats surfaced the noise; understanding *why* removed it.
- **Left a key labelled `trained_100_steps` for a 60-step run.** Caught by
  reading the ledger, not by a test. The step count is now written by the code
  that runs it.

---

## Files

| file | what it is |
|---|---|
| `nanolm.py` | model, masked/chunked/checkpointed cross-entropy, document masks |
| `run_all.py` | every measurement, staged and cached to `results.json` |
| `train_mtp.py` | resumable two-head training |
| `mem_probe.py` | subprocess memory probe (CUDA-exact when a GPU is present) |
| `test_invariants.py` | assertions over every claim above |
| `seed_sweep.py` | resumable multi-seed replication of Part 2 |
| `build_notebook.py` | emits the notebook |
| `results.json` | the numbers ledger |
