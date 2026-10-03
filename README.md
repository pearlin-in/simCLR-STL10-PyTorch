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

*(20-epoch, unscheduled pretraining run — a longer, learning-rate-scheduled 100-epoch run is in progress; see [Status](#status--roadmap).)*

| Labels used | Supervised (scratch) | SSL + linear probe | SSL + full fine-tune |
|---|---|---|---|
| 100% | 68.18% | 67.70% | **76.43%** |
| 10%  | 42.01% | 61.06% | **64.31%** |
| 1%   | 25.12% | **43.44%** | 40.76% |

**Key result:** at 1% of the labels (5 images per class), the frozen SSL encoder beats the fully-trained from-scratch model by **+18.3 points** — despite never seeing a single label during pretraining. The gap is **+19.1 points** at 10% labels. This is the core claim self-supervised learning makes, demonstrated directly rather than assumed.

**A secondary finding worth noting:** at 1% labels, full fine-tuning (40.76%) *underperforms* the frozen linear probe (43.44%). With only 50 training images and an 11M-parameter network fully unfrozen, the model has far more capacity than the data can safely constrain — confirmed by testing a 10x lower learning rate, which underfit instead of fixing the gap (39.31%, train loss stuck at 0.87). This mirrors known findings in transfer-learning research that full fine-tuning can distort pretrained features in low-data regimes, where a frozen probe is the better-suited tool.

## Repository structure

```
.
├── self_supervised_cv.ipynb   # full pipeline: setup → baseline → SSL pretraining → evaluation
├── checkpoints/                # saved encoder weights, probe/finetune checkpoints (gitignored)
├── results.json                 # all measured accuracies, keyed by phase and label %
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

**Done:** supervised baseline, SimCLR pretraining (20-epoch version), linear probing, full fine-tuning, and the fine-tune-vs-probe investigation at 1% labels.

**In progress:** a longer, properly scheduled 100-epoch pretraining run (warmup + cosine learning-rate decay), to be compared against the 20-epoch result as a pretraining-budget ablation.

**Planned:** t-SNE visualization of the learned embedding space, nearest-neighbor retrieval, an augmentation ablation study, and a finer-grained label-efficiency curve (5%, 25% added to the current three points).

## Limitations

- The 20-epoch pretraining results above come from a relatively short, unscheduled run; a longer scheduled run is expected to strengthen the gap further.
- STL-10 is a curated, relatively "easy" dataset compared to the large, uncurated image pools used in production-scale SSL systems — the method generalizes, but these absolute numbers shouldn't be read as representative of that scale.
- All reported test-set numbers are measured once, after training completes, and never used to make training decisions (no early stopping or checkpoint selection on test accuracy).

## References

- Chen et al., ["A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)](https://arxiv.org/abs/2002.05709), 2020.
- Coates, Lee, Ng, ["An Analysis of Single-Layer Networks in Unsupervised Feature Learning"](https://cs.stanford.edu/~acoates/stl10/) (STL-10 dataset).
