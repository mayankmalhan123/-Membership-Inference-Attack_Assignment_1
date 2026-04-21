# Membership Inference Learning Guide

## What The PDF Is Really Asking You To Do

You are not retraining the target model. Your job is to build an **attack model** that predicts whether a sample was in the target model's training set.

The assignment gives you:

- `model.pt`: the trained ResNet-18 target model
- `pub.pt`: a labeled attack-development set with known membership
- `priv.pt`: the hidden-evaluation set where membership is unknown

Your output must be a `submission.csv` with:

- `id`
- `score` in `[0, 1]`

The leaderboard metric is:

- `TPR @ 5% FPR`

That means:

- you only get credit for the top part of your ranking
- false positives are expensive
- calibration matters less than **ranking members above non-members**

## What We Learned From The Public Set

I checked a few classic one-number attacks on `pub.pt` using the provided model:

- Global true-class confidence is almost random: `TPR@5%FPR ~= 0.0507`
- Negative loss is also almost random: `TPR@5%FPR ~= 0.0507`
- Class-conditional ranking helps only a little: `TPR@5%FPR ~= 0.0523`
- A small learned attack model on top of the target model outputs does better on a hold-out split: `TPR@5%FPR ~= 0.0749`

Takeaway:

- a simple threshold on confidence is too weak here
- this task is better treated as **supervised attack learning on `pub.pt`**

## Recommended Workflow

### Step 1: Understand The Objective

Before you code, make sure this sentence is clear in your head:

> "I am using the public set to learn how members and non-members look **through the eyes of the target model**."

That mindset will keep your experiments focused.

### Step 2: Start With A Weak Baseline

Compute one easy score first:

- true-label confidence
- or negative cross-entropy loss

Why do this if it is weak?

- it gives you a sanity check
- it gives you a comparison point for the report
- it helps you explain why stronger attacks are needed

### Step 3: Extract Richer Features

Instead of a single score, build a feature vector from the target model:

- raw logits
- softmax probabilities
- true-label probability
- per-sample loss
- prediction entropy
- correctness
- one-hot label

This is exactly what `guided_attack.py` does.

### Step 4: Train An Attack Model On `pub.pt`

Use `pub.pt` as labeled training data for the attack.

Good first choice:

- a small MLP
- one hidden layer
- binary cross-entropy loss

Why this is better:

- it can learn class-specific behavior
- it can combine weak signals that are useless individually
- it matches the real goal better than hand-picked thresholds

### Step 5: Validate Like A Scientist

Do not trust a method just because it sounds clever.

Split `pub.pt` into train/validation while preserving:

- class label
- membership ratio

Then report:

- `TPR@5%FPR`
- ROC-AUC

Your report becomes much stronger if you can say:

- "Plain confidence was near random."
- "A learned attack on model outputs improved the leaderboard-oriented metric."

### Step 6: Train On All Public Data And Score `priv.pt`

After you settle on the attack setup:

- train on all of `pub.pt`
- extract the same features from `priv.pt`
- output one score per private sample

That is the final submission file.

### Step 7: Iterate Intelligently

The next experiments worth trying are:

- class-specific attack models
- ensembles over multiple random seeds
- extra robustness features from mild perturbations
- calibrating or ranking scores within each class

Do not try everything at once. Change one thing, measure it, keep notes.

## How To Use The Baseline Script

Run a public hold-out evaluation plus generate private scores:

```bash
cd "/Users/mayank.malhan/Documents/Trustworthy ML/tml26_task1"
../.venv-mia/bin/python guided_attack.py --mode both
```

Only evaluate on the public set:

```bash
../.venv-mia/bin/python guided_attack.py --mode eval
```

Only create `submission.csv`:

```bash
../.venv-mia/bin/python guided_attack.py --mode submit
```

## How To Structure The 2-Page Report

### Introduction

Explain in your own words:

- what membership inference is
- why the attacker only sees the trained model and samples
- why `TPR@5%FPR` changes how you design the attack

### Main Body

A clean structure is:

1. Baseline threshold attacks
2. Why they were weak on the public set
3. Learned attack model and chosen features
4. Validation protocol on `pub.pt`
5. Best public leaderboard result

### Conclusion

Answer the practical question:

- if a model leaks membership, what does that mean for privacy?
- why can this matter even when the attacker only gets model outputs?

Strong conclusion ideas:

- training data participation can itself be sensitive
- high-confidence models can still reveal private training inclusion
- membership leakage is a real deployment risk, not just a toy metric
