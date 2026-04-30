# Membership Inference Attack — TML 2026 Assignment 1

This repository contains the code that produced our submission for Task 1 of the Trustworthy Machine Learning 2026 course at CISPA. The submitted attack achieved TPR@5%FPR = 0.056407 on the held-out evaluation set. It is a rank-mean ensemble of three component attacks: an XGBoost stack on shadow-model features (60% weight), a single reference model (25%), and RMIA (15%). The rest of the methodology, including the attacks we tried but did not include in the ensemble, is described in `REPORT_DRAFT.md`.

## Repository contents

The four scripts that, run in sequence, reproduce the final submission:

- `lira_attack.py` — trains 16 shadow ResNet-18 models on random 50% subsets of `pub.pt`, queries them with augmentations, runs LiRA, and (with `--phase score`) also runs an XGBoost stack on the cached shadow features.
- `rmia_attack.py` — RMIA scoring against the shadow cache, with a β sweep and an optional rank-blend against the per-sample shadow z-score.
- `reference_attack.py` — trains one ResNet-18 on `pub.pt` non-members and produces the per-sample `target_logit_conf − reference_logit_conf` gap score.
- `ensemble.py` — rank-mean blender that combines per-attack priv-side scores with manual or auto-derived weights.

Three additional attack scripts produced the negative results reported in §2.4 of the report. Running them is not required to reproduce the final submission, but they are kept here so the report's claims about them are independently verifiable:

- `adversarial_attack.py` — untargeted L_∞ PGD probe at multiple epsilon levels.
- `multi_reference_attack.py` — K reference models on independent non-member subsamples, averaged.
- `query_attack.py` — augmentation- and noise-consistency features blended with RMIA.

`task_template.py` is the file provided with the assignment and is kept verbatim.

## 1. Requirements

The pipeline requires one CUDA-capable GPU with about 8 GB of VRAM and Python 3.10 with PyTorch 2.3, NumPy, pandas, scikit-learn, XGBoost, and LightGBM. The Docker image `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` covers everything except the gradient-boosting libraries:

```bash
pip install xgboost lightgbm scikit-learn
```

## 2. Data

The dataset and target model come from the task's HuggingFace repository. Place all three files in the repository root:

```bash
wget -O pub.pt   https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt
wget -O priv.pt  https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt
wget -O model.pt https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt
```

## 3. Reproducing the submission

The full pipeline runs in roughly an hour on a single GPU. Each step writes its outputs to a deterministic cache directory so subsequent steps can read what earlier ones produced.

### Step 1 — Train 16 shadow models (~50 min)

```bash
python lira_attack.py --phase all --n-shadows 16 --epochs 60 --n-aug 10 --no-stack
```

Trains the 16 shadow ResNet-18s, writes the checkpoints into `shadows/`, the `(16, 14000)` shadow-membership matrix into `shadows/membership.npy`, and per-shadow logit-confidence caches into `lira_cache/shadow_conf_pub.npy` and `lira_cache/shadow_conf_priv.npy`. The `--no-stack` flag suppresses the XGBoost stack at this stage (it is run separately in Step 4).

### Step 2 — Run RMIA on the shadow cache (~5 min)

```bash
python rmia_attack.py --betas 1.0 1.5 2.0 3.0 5.0 --blend-zscore
cp lira_cache/priv_scores_rmia.npy lira_cache/priv_scores_rmia_blend.npy
```

`rmia_attack.py` sweeps β over the supplied values, picks the best on pub TPR@5%FPR, optionally rank-blends with a per-sample z-score against the shadow standard deviation, and writes the per-sample priv scores into `lira_cache/priv_scores_rmia.npy`. The `cp` step gives the file a clearer name for the ensemble step.

### Step 3 — Train and score the reference model (~10 min)

```bash
python reference_attack.py --epochs 60 --n-aug 10
```

Trains one ResNet-18 on the samples in `pub.pt` whose `membership` field equals `0`, queries both that reference and the target with 10 augmentation views each, and writes the gap score `target_logit_conf − reference_logit_conf` into `lira_cache/priv_scores_reference.npy`. The reference checkpoint is saved at `reference/reference.pt`.

### Step 4 — Run the XGBoost stack (~10 min)

```bash
python lira_attack.py --phase score --n-shadows 16
mv submission.csv submission_xgb.csv
```

The `--phase score` invocation reuses the cached shadow data from Step 1 and runs the 5-fold cross-validated XGBoost stack on shadow-derived features (per-sample shadow-confidence statistics, target confidence, normalized gap, and class one-hots). The XGB priv-side predictions are then converted into a NumPy score array for the ensembler:

```bash
python -c "
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
from lira_attack import PRIV_PATH, TaskDataset, MembershipDataset, load_dataset
import __main__
__main__.TaskDataset = TaskDataset
__main__.MembershipDataset = MembershipDataset
df = pd.read_csv('submission_xgb.csv').set_index('id')
ids = [int(x) for x in load_dataset(PRIV_PATH).ids]
np.save('lira_cache/priv_scores_xgb.npy',
        df.loc[ids, 'score'].values.astype(np.float64))
"
```

The two `__main__` assignments are required because `priv.pt` was pickled with `MembershipDataset` rooted in `__main__`; this puts it where Python's unpickler expects.

### Step 5 — Build the final submission

```bash
python ensemble.py \
    --inputs priv_scores_xgb.npy priv_scores_reference.npy priv_scores_rmia_blend.npy \
    --manual-weights 0.6 0.25 0.15 \
    --submission-path submission.csv
```

Each input score array is rank-uniformized to `[0, 1]`, the three are weighted-averaged, and the result is min–max-rescaled. The output `submission.csv` is the file evaluated for our score.

## 4. Other attacks tested

The following scripts produced the negative results reported in the methodology. None is required to reproduce the final submission.

### Adversarial-distance MIA

```bash
python adversarial_attack.py --epsilons 0.05 0.1 0.2 0.3 0.5 --n-steps 10
```

Runs untargeted L_∞ PGD at the supplied epsilon levels and counts, per sample, how many of those perturbations failed to flip the model's prediction. On our target the resulting score gave pub TPR@5%FPR = 0.054 with AUC 0.500 — at the random baseline. The model's adversarial robustness is uncorrelated with training-set membership.

### Multi-reference MIA

```bash
python multi_reference_attack.py --n-refs 6 --subsample-frac 0.7 --epochs 60
```

Trains six reference ResNet-18s on independent random 70% subsamples of the public non-members and averages their `target − reference` gap signals. The averaged signal reached pub TPR ≈ 0.103, only marginally below a single reference's 0.108, indicating that the per-reference overfit is correlated rather than independent across draws.

### Query-only attack

```bash
python query_attack.py --n-aug 40 --noise-levels 0.0 0.05 0.1 0.2 --blend-rmia
```

Queries the target with augmentation views and at four additive Gaussian-noise levels, computes seven per-sample features (mean and standard deviation of the true-class logit confidence, top-2 margin, prediction-mode consistency, noise-vs-confidence slope, mean confidence across noise levels, confidence at the highest noise level), AUC-weight-rank-mean ensembles them, and optionally rank-blends with the cached RMIA score. The combined score reached pub TPR ≈ 0.05 alone or ≈ 0.06 blended with RMIA, weaker than calibrated shadow attacks.

### Augmentation-matched shadow training

```bash
python lira_attack.py --phase all --n-shadows 16 --epochs 60 --aug-shadows \
    --shadow-dir shadows_aug --cache-dir lira_cache_aug --no-stack
python rmia_attack.py --shadow-dir shadows_aug --cache-dir lira_cache_aug \
    --betas 1.0 1.5 2.0 3.0 5.0 --blend-zscore
```

The `--aug-shadows` flag retrains the 16 shadows with random crop and horizontal flip augmentation, on the hypothesis that matching the target's likely augmentation regime would calibrate the IN/OUT shadow distributions better. RMIA on those augmented shadows reached pub TPR 0.0596, slightly worse than the unaugmented baseline of 0.0636.

## 5. Hyperparameters

| Component | Setting |
|---|---|
| Shadow architecture | ResNet-18, custom 3×3 conv1 (no pooling), 9-class output |
| Shadow training | SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, cosine LR over 60 epochs, batch size 128 |
| Shadow training augmentation | None (per the LiRA recommendation) |
| Number of shadows | 16, each on a random 50% subset of `pub.pt` |
| Augmentation views per query | 10 (random crop with reflect-pad, optional horizontal flip) |
| RMIA β values swept | 1.0, 1.5, 2.0, 3.0, 5.0; best chosen on pub TPR@5%FPR |
| Reference model | Same architecture and training as shadows, on `pub_membership = 0` only |
| XGBoost stack | 5-fold class- and membership-stratified CV, n_estimators=2000, max_depth=6, lr=0.03, early stopping=50 |
| Ensemble weights | 0.60 XGB / 0.25 reference / 0.15 RMIA |
| Ensemble combiner | Per-attack rank-uniformization, weighted average, min–max rescale to `[0, 1]` |

## 6. File layout

```
lira_attack.py              # shadow training, LiRA, XGB stack
rmia_attack.py              # RMIA on cached shadow confidences
reference_attack.py         # single reference model + gap score
ensemble.py                 # rank-mean blender
adversarial_attack.py       # PGD distance probe
multi_reference_attack.py   # K reference models averaged
query_attack.py             # augmentation + noise consistency probe
task_template.py            # provided by the course
README.md                   # this file
REPORT_DRAFT.md             # the report submitted to CMS
shadows/                    # shadow checkpoints + membership matrix (gitignored)
reference/                  # reference checkpoint (gitignored)
multi_reference/            # multi-reference checkpoints (gitignored)
lira_cache/                 # shadow + target queries, per-attack scores (gitignored)
submission.csv              # final attack output (gitignored)
```

## 7. Reproducibility note

All seeds default to 0. The pipeline is deterministic up to CUDA/cuDNN nondeterminism in shadow training, which produces small differences in the per-shadow weights between runs. End-to-end the final score reproduces to roughly ±0.001.
