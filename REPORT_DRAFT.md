# Membership Inference Attack on a ResNet-18 Image Classifier

GitHub repository URL: `REPLACE_WITH_YOUR_GITHUB_REPO_URL`

## Introduction

This task studies membership inference against a trained image classification model. The attacker receives a pretrained ResNet-18, a public dataset with known membership labels, and a private dataset where membership is hidden. The goal is to assign each private sample a score between 0 and 1, where higher scores indicate that the sample is more likely to have appeared in the target model's training set. Since the evaluation metric is `TPR@5%FPR`, the attack should not only separate members from non-members, but should do so while keeping false positives low. This makes ranking quality more important than simple accuracy.

## Main Body

### Approach

I treated the task as a supervised attack-learning problem. Instead of retraining the target classifier, I used `pub.pt` as labeled data for training a separate attack model. The main idea is that members and non-members may look slightly different through the outputs of the target model, even when they come from the same underlying data distribution.

I first evaluated simple threshold-based attacks, such as ranking samples by true-label confidence or by negative cross-entropy loss. These baselines are standard in membership inference because training samples often produce lower loss and higher confidence. However, in this assignment these single-score attacks were weak on the public set and behaved close to random ranking. This suggested that the leakage signal exists, but is subtle and not well captured by a single handcrafted statistic.

### Feature Design And Attack Model

To capture richer leakage signals, I extracted a feature vector from the target model for every sample. The feature vector includes raw logits, softmax probabilities, true-label confidence, per-sample loss, prediction entropy, correctness, and the one-hot encoding of the ground-truth class. These features combine calibration information, uncertainty, and class-specific behavior in a way that a learned attack model can exploit.

On top of these features, I trained a small multilayer perceptron with one hidden layer and a binary cross-entropy objective. I kept the architecture intentionally simple in order to reduce overfitting on the public set and to maintain a clear connection between the chosen features and the final membership score. Input features were standardized using the training split statistics before fitting the attack model.

### Validation And Results

To validate the method, I split `pub.pt` into attack-train and attack-validation subsets while preserving both class balance and the member/non-member ratio within each class. I then measured performance using the same leaderboard-oriented metric as the assignment, namely `TPR@5%FPR`, and also tracked ROC-AUC as a secondary ranking metric.

The strongest simple baselines, such as true-label confidence and negative loss, achieved only about `0.0507` TPR at `5%` FPR on the public data, which is close to random performance. In contrast, the learned attack improved the validation score to about `0.0756` TPR at `5%` FPR, with a ROC-AUC of about `0.5246`. This result shows that the target model leaks some membership information, but that the leakage is weak and requires combining multiple output-based signals.

Public leaderboard score: `REPLACE_WITH_YOUR_BEST_PUBLIC_LEADERBOARD_SCORE`

## Conclusion

This assignment shows that privacy leakage can persist even when the attacker only observes a trained model and candidate samples. A successful membership inference attack means that participation in the training set can itself become sensitive information. In practice, this matters whenever the presence of a record in training reveals something private about an individual, such as medical status, platform usage, or membership in a protected group. Even though the attack in this task is not extremely strong, it demonstrates that nontrivial privacy leakage can still arise from standard predictive models, which is exactly why model auditing and privacy-aware training methods are important.
