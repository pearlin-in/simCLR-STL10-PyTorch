# Project Write-Up: Self-Supervised Pretraining for Label-Efficient Image Classification

A complete record of the process, decisions, numerics, and issues encountered while building and evaluating a SimCLR contrastive pretraining pipeline on STL-10 — written as a running log rather than a polished summary, so the reasoning behind each change is visible.

---

## 1. Objective

Measure how much contrastive self-supervised pretraining (SimCLR) improves label efficiency versus training a classifier from scratch, at 100%, 10%, and 1% of the available labeled data, on STL-10.

Three conditions are compared at each label fraction:
- **Supervised from scratch** — random-init ResNet-18, trained directly on labels.
- **Linear probe** — SSL-pretrained encoder frozen, only a new linear layer trained.
- **Full fine-tune** — SSL-pretrained encoder unfrozen, entire network trained on labels.

---

## 2. Build order

1. Environment setup, STL-10 loading (labeled train / unlabeled / test splits).
2. Supervised baseline at 100%/10%/1% labels (the number SSL has to beat).
3. SimCLR augmentation pipeline + contrastive data loader (positive pairs from the 100k unlabeled images).
4. SimCLR encoder + projection head, NT-Xent loss (implemented and independently verified against a slow loop-based reference), pretraining loop.
5. Linear probing at 100%/10%/1% labels.
6. Full fine-tuning at 100%/10%/1% labels.
7. A second, improved pretraining run (scheduled, longer) once the first pass proved the method worked — repeating steps 5–6 against the new encoder.

---

## 3. Issues found and fixes made, in the order they came up

### 3.1 — Fine-tune encoder loading crashed
`make_finetune_model()` initially loaded the entire checkpoint dictionary (`{'epoch', 'model_state_dict', 'optimizer_state_dict', ...}`) directly into the encoder instead of extracting `model_state_dict` and stripping the `"encoder."` key prefix — the way the linear-probe loader already did correctly. Fixed by mirroring that logic exactly, and loading with `strict=True` so a bad load fails loudly instead of silently.

### 3.2 — Dead filter logic in the probe loader
A key filter checked for `"projector."`, but the model's actual attribute is `projection_head`, so the filter never matched anything. Didn't break the run (`strict=False` absorbed it), but meant the safety check wasn't doing its job. Fixed by explicitly printing `missing`/`unexpected` keys after every encoder load, so a genuinely broken load is now visible. Confirmed clean on every run since: `missing: []`, `unexpected: [the four projection-head keys, correctly discarded]`.

### 3.3 — Results lost between Colab sessions
Accuracy numbers only existed as in-memory Python variables; every free-tier disconnect wiped them, forcing retraining just to regenerate a single percentage. Fixed by adding `save_result()` / `results.json`, writing every computed accuracy to Drive immediately.

### 3.4 — First pretraining pass was short and unscheduled
The first SimCLR run used 20 epochs at a constant learning rate — well below typical SimCLR convergence. It still produced a real, usable result, but left clear room for improvement (see Section 4).

### 3.5 — Checkpoint format broke across the scheduler upgrade
Adding a learning-rate scheduler meant checkpoints needed a new `scheduler_state_dict` key. Loading an old-format checkpoint into the new code threw `KeyError: 'scheduler_state_dict'`. Resolved by archiving the old 20-epoch checkpoint under a separate filename (kept deliberately, not deleted, for the ablation in Section 5) and making the archive step conditional on actually detecting the old format — so re-running the same setup cell on a later session could never accidentally re-archive and wipe out in-progress 100-epoch training.

### 3.6 — Non-atomic checkpoint writes
The original save wrote directly to the final checkpoint filename. A runtime kill mid-write (a real risk on free-tier Colab) could have left a truncated, unloadable file, destroying hours of progress. Fixed with a temp-file-then-rename (atomic) write pattern.

### 3.7 — Stale downstream checkpoints after the encoder changed
After the 100-epoch encoder finished, re-running the probe/fine-tune cells risked silently resuming from `probe_100.pth` / `finetune_100.pth` files left over from the *old* 20-epoch encoder's runs. Since those runs had already completed all their epochs, the resume logic would skip training entirely and report a stale, meaningless number — with no error. Fixed by explicitly archiving all old probe/fine-tune checkpoints before rerunning Phase 4/5 against the new encoder.

### 3.8 — GPU quota exhausted mid-run (free tier)
Partway through the 100-epoch run, free-tier Colab's GPU quota ran out (`AssertionError: Torch not compiled with CUDA enabled` after reconnecting on a CPU-only runtime). Not a code issue — resolved by waiting for the quota to reset and resuming via the existing checkpoint, which worked exactly as designed.

### 3.9 — Run-to-run variance in small-data experiments
The supervised baseline and full fine-tune cells retrain from a new random initialization every time they're run (no fixed seed), and at 50–500 training images, this produces a few points of accuracy variance between otherwise-identical reruns. Not a bug, but worth stating plainly: a single run's number at 1%/10% labels is a sample, not an exact constant. (Noted as a possible future improvement — see Section 7.)

---

## 4. Why the pretraining was redone: 20 epochs vs. 100 epochs

The first SimCLR run (20 epochs, constant learning rate) already produced a usable positive result, but was a deliberately conservative first pass. It was redone with:
- **100 epochs** instead of 20.
- **Learning-rate warmup** (10 epochs ramping up) followed by **cosine decay** across the full run, instead of a flat constant rate.
- The same checkpoint/resume infrastructure, extended to also persist and restore scheduler state, so the much longer run could survive free-tier Colab's session limits across several days.

This was treated as a deliberate ablation rather than simply "redoing it better" — both encoders were evaluated independently and kept, specifically to measure how much pretraining budget actually mattered (Section 5).

---

## 5. Results

### 5.1 — Supervised baseline vs. SSL, 20-epoch encoder (first pretraining run)

| Labels | Supervised (scratch) | SSL + linear probe | SSL + full fine-tune |
|---|---|---|---|
| 100% | 68.18% | 67.70% | 76.43% |
| 10% | 42.01% | 61.06% | 64.31% |
| 1% | 25.12% | 43.44% | 40.76% (39.31% at a 10x lower fine-tune learning rate) |

Gap at 1% labels (best SSL result vs. scratch): **+18.3 points**.

### 5.2 — Supervised baseline vs. SSL, 100-epoch scheduled encoder (final pretraining run)

Final, authoritative numbers, confirmed directly from `results.json`:

| Labels | Supervised (scratch) | SSL + linear probe | SSL + full fine-tune |
|---|---|---|---|
| 100% | 69.34% | 75.40% | **81.28%** |
| 10% | 38.95% | 69.29% | **70.34%** |
| 1% | 24.39% | **50.29%** | 46.88% (42.74% at a 10x lower fine-tune learning rate) |

Gap at 1% labels (best SSL result vs. scratch): **+25.9 points**. Gap at 10% labels (best SSL result vs. scratch): **+31.4 points**. Both meaningfully stronger than the first pretraining run.

Note on the scratch baseline at 10% labels (38.95%): this is noticeably lower than figures seen in earlier reruns of this same cell (39–43% range across different runs), consistent with the random-initialization variance discussed in Section 3.9 rather than any change in method. This is the final value actually saved and used for all comparisons above.

### 5.3 — Pretraining-budget ablation (linear probe, encoder-only comparison — not affected by scratch/fine-tune run-to-run variance)

| Labels | 20-epoch encoder | 100-epoch encoder | Improvement |
|---|---|---|---|
| 100% | 67.70% | 75.40% | +7.7 pts |
| 10% | 61.06% | 69.29% | +8.2 pts |
| 1% | 43.44% | 50.29% | +6.9 pts |

A consistent, substantial improvement across every label fraction, with no sign of plateauing — the extra pretraining budget directly translated into stronger downstream features.

### 5.4 — Fine-tuning vs. linear probing: the full picture across both encoders

| Encoder | Frozen linear probe | Full fine-tune (standard LR) | Full fine-tune (10x lower LR) |
|---|---|---|---|
| 20-epoch, 1% labels | 43.44% | 40.76% | 39.31% |
| 100-epoch, 1% labels | 50.29% | 46.88% | 42.74% |

The frozen probe beats full fine-tuning at 1% labels in **every single configuration tested**, across two independently trained encoders and two different fine-tuning learning rates. Lowering the learning rate does not close the gap — it makes it worse (underfitting instead of overfitting), confirming this is a capacity-vs-data-size mismatch (an 11M-parameter network fully unfrozen, trained on just 50 images) rather than a tunable hyperparameter issue.

**How this gap changed with more pretraining, at every label fraction** (finetune accuracy minus probe accuracy; positive means fine-tuning wins):

| Labels | 20-epoch encoder | 100-epoch encoder | Change |
|---|---|---|---|
| 100% | +8.73 pts | +5.88 pts | gap narrowed |
| 10% | +3.25 pts | +1.05 pts | gap narrowed |
| 1% | **−2.68 pts** | **−3.41 pts** | gap widened slightly |

The pattern is not simply "better pretraining fixes fine-tuning's low-data weakness." At 10% and 100% labels, fine-tuning's advantage over probing shrank as pretraining improved — the frozen probe got close enough that full fine-tuning's extra adaptation mattered less. But at the true data-starved extreme (1% labels, 50 images), a stronger, richer encoder didn't rescue fine-tuning from the pattern — if anything, richer features may give an unfrozen network more to overfit around with so little data to constrain it. This is a more precise and more interesting conclusion than "more pretraining always helps everything," and it's the kind of nuance that only shows up by actually running the full matrix of experiments rather than assuming the result.

---

## 6. Interpretation

- **The core self-supervised learning claim holds, and holds more strongly with more pretraining**: a model that never saw a single label beats a fully-labeled, from-scratch classifier by 18–31 points when labels are scarce, with the gap widening as the SSL pretraining budget increases.
- **Full fine-tuning is not strictly better than linear probing, and the relationship is more nuanced than "more pretraining fixes it."** Fine-tuning is the better choice once there's enough labeled data (clear, and growing more efficient, wins at 10% and 100% labels as pretraining improved). But at 1% labels, frozen linear probing beat fine-tuning with *both* encoders, and the gap didn't close with more pretraining — it widened slightly (−2.68 → −3.41 points). A richer encoder helps fine-tuning close ground once there's a moderate amount of data, but doesn't rescue it from overfitting at the true data-starved extreme. This mirrors known results in transfer-learning research on feature distortion during fine-tuning in low-data regimes.
- **Pretraining duration/schedule matters substantially**, not just as a minor tweak: moving from 20 unscheduled epochs to 100 scheduled epochs (warmup + cosine decay) improved every single downstream number, with gains that hadn't visibly plateaued by epoch 100 — suggesting results would likely improve further with an even longer budget, compute permitting.
- **The t-SNE visualization provides direct visual confirmation** of what the accuracy numbers imply: embeddings from the SSL-pretrained encoder show clear, semantically sensible clustering by true class (vehicles — car, truck, ship — group separately from animals; visually similar animal classes like dog/cat/horse/monkey/deer show expected partial overlap) despite the encoder never seeing a single label. The random-init control encoder, evaluated identically, shows no structure at all — confirming the clustering is a genuine product of pretraining, not an artifact of the visualization method itself.

---

## 7. Known limitations

- STL-10 is a curated, relatively "easy" dataset compared to the large, uncurated image pools used in production-scale SSL systems; the method generalizes, but absolute numbers here shouldn't be read as representative of that scale.
- Scratch-baseline and fine-tune results carry several points of run-to-run variance due to unseeded random initialization at small data sizes — a fixed random seed (or averaging over multiple runs) would make these numbers more rigorously reproducible, and is a reasonable next improvement if this project is extended.
- All reported test-set numbers are measured once after training completes and never used to make training decisions (no early stopping or checkpoint selection on test accuracy).
- 100 epochs of pretraining was a practical stopping point dictated by free-tier Colab compute limits, not a point of observed diminishing returns.

---

## 8. Status

**Complete:** supervised baseline, SimCLR pretraining (both 20-epoch and 100-epoch scheduled versions), linear probing and full fine-tuning against both encoders, the fine-tune-vs-probe investigation at 1% labels (replicated across both encoders and two learning rates), the pretraining-budget ablation, and the t-SNE visualization against the finished 100-epoch encoder. All numerics in this document are confirmed directly from `results.json`.

**Open (optional):** nearest-neighbor retrieval, an augmentation ablation study (retraining with one augmentation removed), and a finer-grained label-efficiency curve (5%, 25% added to the current three points).