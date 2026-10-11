# Self-Supervised Computer Vision: Label-Efficient Image Classification

Pretraining a ResNet-18 with contrastive self-supervised learning (SimCLR) on unlabeled images, then measuring how much that pretraining reduces the need for labeled data — compared directly against a from-scratch supervised baseline.

## Why this project

Labeled data is expensive; unlabeled data is everywhere. Self-supervised learning asks whether a model can learn useful visual features with zero labels, by solving a made-up task invented from the data itself — and whether those features then make the model dramatically more label-efficient on a real downstream task. This project implements that pipeline end-to-end and measures the effect directly, rather than just citing the claim.

## Method

- **Pretraining task (SimCLR):** for each unlabeled image, generate two independently augmented views (random resized crop, horizontal flip, color jitter, grayscale, Gaussian blur). Train an encoder so the two views of the same image produce similar embeddings, while views of different images are pushed apart.
- **Loss:** NT-Xent (normalized temperature-scaled cross-entropy), implemented as a vectorized similarity-matrix computation and independently verified against a slow loop-based reference implementation (exact match to 1e-5).
- **Architecture:** ResNet-18 encoder (512-dim output) + a 2-layer MLP projection head used only during pretraining and discarded afterward, per the SimCLR paper.
- **Evaluation protocol:** compare three training regimes at 100%, 10%, and 1% of the labeled data:
  1. **Supervised from scratch** — random-init ResNet-18, trained directly on labels.
  2. **Linear probe** — SSL-pretrained encoder frozen, only a new linear layer trained on labels.
  3. **Full fine-tune** — SSL-pretrained encoder unfrozen, entire network trained on labels.

## Dataset

[STL-10](https://cs.stanford.edu/~acoates/stl10/): 5,000 labeled train images, 8,000 labeled test images, 100,000 unlabeled images (96×96, 10 classes). The unlabeled split is used only for pretraining; labels are never touched until the evaluation phases.

## Results

| Labels used | Supervised (scratch) | SSL + linear probe | SSL + full fine-tune |
|---|---|---|---|
| 100% | 67.98% | 75.40% | **80.46%** |
| 10%  | 42.76% | **69.29%** | 69.63% |
| 1%   | 25.14% | **50.29%** | 49.24% |

**Key result:** at 1% of the labels (5 images per class), the SSL-pretrained encoder beats the fully-trained from-scratch model by **+25.2 points** — despite never seeing a single label during pretraining. The gap is **+26.5 points** at 10% labels. This is the core claim self-supervised learning makes, demonstrated directly rather than assumed.

### Pretraining-budget ablation: does longer, scheduled pretraining actually help?

Two full encoders were trained and evaluated end-to-end: a 20-epoch run with a constant learning rate, and a 100-epoch run with warmup + cosine learning-rate decay.

| Labels | Linear probe, 20-epoch SSL | Linear probe, 100-epoch SSL | Improvement |
|---|---|---|---|
| 100% | 67.70% | 75.40% | +7.7 pts |
| 10%  | 61.06% | 69.29% | +8.2 pts |
| 1%   | 43.44% | 50.29% | +6.9 pts |

Longer, properly-scheduled pretraining produced a substantial, consistent improvement across every label fraction — pretraining duration and schedule materially affect downstream performance, not just a marginal tweak.

### When does full fine-tuning underperform linear probing?

At 1% labels, full fine-tuning underperformed the frozen linear probe in **both** pretraining runs, but the gap shrank as pretraining improved:

| Labels | Finetune − probe, 20-epoch SSL | Finetune − probe, 100-epoch SSL |
|---|---|---|
| 100% | +8.73 pts | +5.06 pts |
| 10%  | +3.25 pts | +0.34 pts |
| 1%   | **−2.68 pts** | **−1.05 pts** |

With only 50 training images and an 11M-parameter network fully unfrozen, the model has far more capacity than the data can safely constrain — confirmed in the 20-epoch run by testing a 10x lower learning rate, which underfit instead of closing the gap (39.31%, train loss stuck at 0.87, vs. 40.76% at the standard rate). A stronger pretrained encoder partially mitigates this effect but does not eliminate it, even at a 100-epoch pretraining budget. This mirrors known findings in transfer-learning research that full fine-tuning can distort pretrained features in low-data regimes, where a frozen probe is the better-suited tool.

## Repository structure

```
.
├── self_supervised_cv.ipynb   # full pipeline: setup → baseline → SSL pretraining → evaluation
├── need to update :))               
└── figures/
    ├── label_efficiency_curve.png
    └── tsne_comparison.png
```

## Running it

Built and run on Google Colab (free-tier T4 GPU). Requires PyTorch + torchvision (preinstalled on Colab).

1. Open `self_supervised_cv.ipynb` in Colab, mount Google Drive.
2. Run cells top to bottom through Phase 1 for the supervised baseline.
3. Run Phase 2–3 to build the augmentation pipeline and pretrain the SimCLR encoder. This is the most compute-intensive step (~13 min/epoch on a T4); the training loop checkpoints every 50 steps and resumes automatically across disconnected sessions.
4. Run Phase 4–5 to evaluate the pretrained encoder via linear probing and full fine-tuning.

## Status & roadmap

**Done:** supervised baseline, SimCLR pretraining (both a 20-epoch unscheduled run and a 100-epoch warmup+cosine scheduled run), linear probing and full fine-tuning against both encoders, the fine-tune-vs-probe investigation at 1% labels, and the pretraining-budget ablation comparing the two runs.

**Planned:** t-SNE visualization of the learned embedding space (against the 100-epoch encoder), nearest-neighbor retrieval, an augmentation ablation study, and a finer-grained label-efficiency curve (5%, 25% added to the current three points).

## Limitations

- STL-10 is a curated, relatively "easy" dataset compared to the large, uncurated image pools used in production-scale SSL systems — the method generalizes, but these absolute numbers shouldn't be read as representative of that scale.
- All reported test-set numbers are measured once, after training completes, and never used to make training decisions (no early stopping or checkpoint selection on test accuracy).
- Pretraining duration clearly still matters at 100 epochs (see the ablation above) — the gains hadn't visibly plateaued, so an even longer run would likely improve results further; 100 epochs was chosen as a practical stopping point given free-tier compute constraints, not a point of diminishing returns.

## References

- Chen et al., ["A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)](https://arxiv.org/abs/2002.05709), 2020.
- Coates, Lee, Ng, ["An Analysis of Single-Layer Networks in Unsupervised Feature Learning"](https://cs.stanford.edu/~acoates/stl10/) (STL-10 dataset).
