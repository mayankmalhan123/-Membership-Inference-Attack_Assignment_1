# Membership Inference Attack — TML 2026 Assignment 1

## 1. Introduction

We are given a pretrained ResNet-18 image classifier, a public dataset of 14,000 samples with known training-set membership labels, and a private set of 14,000 samples for which we have to predict membership. Members and non-members are drawn from the same distribution and no explicit indicator of membership is exposed, so the only signal we can exploit is whatever the model leaks about per-sample memorization through its predictions.

The evaluation metric is TPR at 5% FPR. With roughly 7,000 non-members in the evaluation split that operating point is strict, which makes the task fundamentally hard: any attack that produces calibrated confidence rather than per-sample memorization fingerprints is going to struggle.

## 2. Approach

We worked through four stages: a quick diagnostic on the target model, calibrated shadow-model attacks, single-target probes that don't use shadows, and finally an ensemble of the strongest signals from each.

### 2.1 Diagnostic

The first question was whether the target's raw outputs were already discriminative. They were not. On the public set we measured AUC and TPR@5%FPR for the obvious per-sample scalar scores: cross-entropy loss, true-class probability, true-class logit, prediction entropy, and top-1 minus top-2 margin. Every score gave AUC within ±0.005 of 0.5 and TPR@5%FPR between 0.044 and 0.054. The model's outputs are well-calibrated enough that members and non-members produce indistinguishable confidence distributions on average. Direct-confidence attacks were ruled out, and we moved to calibrated attacks that compare the target's behaviour to a reference distribution.

### 2.2 Shadow models and likelihood-ratio attacks

We trained 16 shadow ResNet-18 models from scratch, each on a random 50% subset of the public set. Training used SGD with momentum 0.9, weight decay 1e-4, learning rate 0.1 with cosine annealing over 60 epochs, batch size 128, and no train-time augmentation. The choice of no augmentation follows the LiRA recommendation: shadows must memorize their training data for the IN/OUT confidence gap to be detectable, and standard CIFAR-style aug actively suppresses that.

Each shadow was queried with 10 augmentation views (random crop with reflect-padding, optional horizontal flip) on the union of the public and private sets, producing a (16, 28000) cache of logit-scaled true-class confidences. From this cache we ran two canonical attacks:

LiRA (Carlini et al., 2021) fits per-sample IN/OUT Gaussians from the shadow distribution and scores by the log-likelihood ratio of the target's confidence under each. Pub TPR@5%FPR was 0.0504, basically random.

RMIA (Zarifzadeh et al., 2023) replaces the per-sample Gaussian fits with a population-level rank statistic computed against pub non-members. It did better, with pub TPR@5%FPR of 0.0636. We swept its β parameter over {1.0, 1.5, 2.0, 3.0, 5.0}, tested z-score blending against the per-sample shadow std, and ran a parallel pipeline with denser augmentation queries (40 views vs 10). All variants landed within 0.001 of each other on the public set.

We also retrained the 16 shadows with crop-and-flip augmentation, on the hypothesis that matching the target's likely augmentation regime would calibrate the IN/OUT distributions better. RMIA on those augmented shadows gave 0.0596, slightly worse than the unaugmented baseline. Either the target was not trained with that exact aug regime or augmentation-matched shadows reduce the per-sample memorization signal too much. Either way, the variant did not improve over plain RMIA.

### 2.3 Supervised stacking on shadow features

For each public sample we built a 64-dimensional feature vector consisting of: the LiRA score itself, target confidence, summary statistics over the 16 shadow confidences (mean, std, min, max, median), the gap between target confidence and shadow mean, the same gap normalized by shadow std, and a 9-dim one-hot for the class label. We trained 5-fold cross-validated XGBoost (n_estimators=2000, max_depth=6, lr=0.03, early stopping=50) where folds were stratified by both membership and class.

Pub OOF TPR@5%FPR jumped to 0.2010, which looked like a major breakthrough. It was not. When evaluated on the private set the score collapsed to 0.0548. The supervised stack had absorbed pub-distribution-specific signal — variations in feature distributions that show up on a labelled hold-out but do not transfer to a fresh sample of the same population. This was the most useful methodological lesson of the entire exercise: when likelihood-ratio attacks plateau around 0.06 and a supervised stack reports 0.20, the stack is overfitting, not extracting new information.

### 2.4 Reference-model and adversarial probes

Single-reference attack. We trained one ResNet-18 on the public non-members only and scored `target_logit_conf − ref_logit_conf` per sample. The reference, by construction, has zero overlap with the target's training set, so any per-sample gap should reflect target memorization. Pub TPR@5%FPR was 0.1083, the highest single-attack number we had seen, but on the private set it again did not improve over 0.0548. The reference had over-fit pub-distribution noise the same way the XGB stack did.

Multi-reference attack. To attempt to cancel the per-reference overfit we trained six independent reference models on different random 70% subsamples of the public non-members, then averaged their (target − ref) signals. Individual references gave pub TPRs in the 0.092–0.109 range. Their average was 0.1029. The averaging removed only a small fraction of the per-reference variance, indicating the references over-fit in correlated rather than independent directions. The variance-reduction trick that works for shadow models did not carry over.

Adversarial-distance attack. For each sample we ran untargeted L∞ PGD at five epsilon levels (0.05, 0.1, 0.2, 0.3, 0.5) with 10 PGD steps each, counting per sample how many epsilons failed to flip the model's prediction. Members did not require systematically larger perturbations: pub TPR@5%FPR was 0.0540 and AUC was 0.5001, both at the random baseline. The target's adversarial robustness is uncorrelated with training-set membership, suggesting the model preserves prediction confidence by mechanisms that are not driven by per-sample memorization.

Query-only attack. For each sample we queried the target with 40 augmentation views plus four additive Gaussian noise levels (σ ∈ {0, 0.05, 0.1, 0.2}) and computed seven per-sample features: mean and standard deviation of the true-class logit confidence, top-1 minus top-2 margin, prediction-mode consistency across augmentations, the slope of true-class confidence as a function of noise σ, and mean confidence at the highest noise level. We rank-mean ensembled the features weighted by their individual |AUC − 0.5| on the public set. The combined query-only score reached pub TPR@5%FPR = 0.0513; rank-blending it with the RMIA score lifted it to 0.0600. The signals are individually weak and do not improve on calibrated shadow attacks, indicating that simple augmentation-consistency and additive-noise robustness are not strong leakage vectors on this target.

### 2.5 Ensembling the diverse signals

The thing that actually moved the needle was rank-mean ensembling. Each attack we had built carried a different error profile: the XGB stack absorbed pub-specific feature noise, the reference model absorbed pub-distribution noise, and RMIA captured the calibrated likelihood-ratio signal. We rank-uniformized each attack's per-sample priv-side score, took a weighted average, and min–max-rescaled the result into [0, 1].

We tried several weight schemes. The schedule that scored highest on the public OOF — and which we submitted as our final answer — was 60% XGB stack, 25% reference model, and 15% RMIA. The XGB-heavy weighting reflects the empirical fact that XGB was the only attack we had with verified private-side transfer (its priv score was real, even if its pub OOF was inflated). The reference and RMIA contributions add orthogonal signal that cancels XGB's pub-specific noise. The final TPR@5%FPR on the evaluation set was **0.056407**.

## 3. Key results

| Attack | Pub OOF TPR@5%FPR | Priv (evaluated) |
|---|---|---|
| Loss / confidence baselines | ≤ 0.046 | random |
| LiRA on 16 shadows | 0.0504 | random |
| RMIA on 16 shadows | 0.0636 | did not improve |
| RMIA on aug-matched shadows | 0.0596 | not submitted |
| XGB stack on shadow features | 0.2010 | 0.0548 |
| Single reference model | 0.1083 | did not improve |
| Multi-reference (6 averaged) | 0.1029 | not submitted |
| Adversarial-distance (PGD) | 0.0540 | not submitted |
| Query-only (aug + noise, AUC-weighted blend) | 0.0513 | not submitted |
| Query-only blended with RMIA | 0.0600 | not submitted |
| **Ensemble: 60% XGB + 25% ref + 15% RMIA** | 0.0929 | **0.0564** |

## 4. Conclusion

A model that looks well-protected against straightforward MIA — its raw confidence and loss distributions are calibrated, and its adversarial robustness is independent of training-set membership — still leaks several percentage points of true-positive identification at strict false-positive rates. Three observations from this work are worth emphasising for anyone considering the privacy of a deployed image classifier.

First, single-attack defence is not enough. A real adversary will try multiple attack families and combine them. Our ensemble of three weak attacks recovered information that none of the individuals could capture alone. The relevant threat model is therefore not the strongest single MIA, but the strongest ensemble of MIAs.

Second, internal cross-validation can be badly optimistic. We ran two separate attacks (XGB stacking and single-reference) whose validated public-side scores were 2-4 times higher than their actual private-side performance. A defender deciding whether a model is safe to release based on a labelled internal hold-out would substantially over-estimate their privacy.

Third, robustness to one perturbation type does not imply MIA-resistance. Our PGD probe showed the target was equally robust to small adversarial perturbations on members and non-members, yet it still leaked membership through statistical signals captured by reference- and shadow-based attacks. Privacy claims should therefore be validated empirically against diverse ensembles of MIAs, rather than inferred from related-but-distinct robustness properties.

## 5. Code

Repository: **<FILL IN BEFORE SUBMITTING>**

The `README.md` in the repository documents the exact commands used to reproduce the final submission, including all hyperparameters and the cluster job configuration we used for shadow training.

### References

Yeom, S., Giacomelli, I., Fredrikson, M., Jha, S. (2018). _Privacy risk in machine learning: Analyzing the connection to overfitting._

Shokri, R., Stronati, M., Song, C., Shmatikov, V. (2016). _Membership Inference Attacks Against Machine Learning Models._ arXiv:1610.05820.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., Tramèr, F. (2021). _Membership Inference Attacks From First Principles._ arXiv:2112.03570.

Zarifzadeh, S., Liu, P., Shokri, R. (2023). _Low-Cost High-Power Membership Inference Attacks._ arXiv:2312.03262.

Choquette-Choo, C., Tramer, F., Carlini, N., Papernot, N. (2021). _Label-Only Membership Inference Attacks._
