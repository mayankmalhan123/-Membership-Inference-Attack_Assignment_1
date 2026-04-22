# Membership Inference Attack on a ResNet-18 Image Classifier

GitHub repository URL: `https://github.com/mayankmalhan123/-Membership-Inference-Attack_Assignment_1`

## Introduction

This task studies membership inference against a trained image classification model. The attacker receives a pretrained ResNet-18, a public dataset with known membership labels, and a private dataset where membership is hidden. The objective is to assign each private sample a score in `[0,1]`, where higher scores indicate a higher likelihood that the sample was part of the target model's training set. Because the evaluation metric is `TPR@5%FPR`, success depends less on overall accuracy and more on producing a strong ranking while keeping false positives very low.

## Main Body

### Approach

I approached the problem as a supervised attack-learning task. Instead of retraining the target classifier, I used `pub.pt` as labeled data for a separate attack model. The central assumption is that members and non-members may induce slightly different output patterns in the target model, even when both are drawn from the same underlying data distribution.

I first tested simple threshold-based attacks, such as ranking samples by true-label confidence or by negative cross-entropy loss. These baselines are standard in membership inference because training samples often yield lower loss and higher confidence. However, in this assignment they performed close to random on the public data. This indicated that any privacy leakage was likely weak and not well captured by a single handcrafted score.

### Feature Design And Attack Model

To capture richer leakage signals, I extracted a feature vector from the target model for every sample. The feature vector includes raw logits, softmax probabilities, true-label confidence, per-sample loss, prediction entropy, correctness, and the one-hot encoding of the ground-truth class. Together, these features describe confidence, calibration, uncertainty, and class-specific behavior in a form that a learned attack model can use.

On top of these features, I trained a small multilayer perceptron with one hidden layer and a binary cross-entropy objective. I intentionally kept the architecture simple to limit overfitting on the public set and to maintain a clear connection between the extracted features and the final membership score. Before training, I standardized the input features using statistics computed on the attack-training split.

### Validation And Results

To validate the method, I split `pub.pt` into attack-train and attack-validation subsets while preserving both class balance and the member/non-member ratio within each class. I measured performance using the assignment metric, `TPR@5%FPR`, and also tracked ROC-AUC as a secondary ranking measure.

The strongest simple baselines, such as true-label confidence and negative loss, achieved only about `0.0507` TPR at `5%` FPR on the public data, which is close to random performance. In contrast, the learned attack improved the local validation score to about `0.0756` TPR at `5%` FPR, with a ROC-AUC of about `0.5246`. This suggests that the target model does leak some membership information, but the signal is weak and only becomes visible when several output-based cues are combined.

My public leaderboard score was `0.047823`. This is lower than the local hold-out estimate and slightly below the near-random `0.05` level for this metric. A likely explanation is mild overfitting to the public development split together with weak generalization to the hidden leaderboard subset of `priv.pt`. Because `TPR@5%FPR` depends only on the extreme top of the ranking, even a small shift in ordering between local validation samples and hidden evaluation samples can noticeably reduce the final score. This result is still informative: it suggests that the leakage signal is weak, unstable, and sensitive to distribution shift, which makes careful validation and conservative interpretation especially important.

## Conclusion

This assignment shows that privacy leakage can persist even when the attacker only has access to a trained model and a set of candidate samples. A successful membership inference attack means that mere participation in the training set can become sensitive information. In practice, this matters whenever inclusion in training reveals something private about an individual, such as medical status, platform usage, or membership in a protected group. Even though the attack in this task was not strong on the public leaderboard, it still demonstrates that standard predictive models can leak nontrivial information about their training data. This is precisely why privacy auditing and privacy-aware training methods are important components of trustworthy machine learning.
