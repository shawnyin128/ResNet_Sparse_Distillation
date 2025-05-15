we propose a distillation method that enables the student model to learn both weight
and activation sparsity directly during training, by incorporating L1 regularization and soft KL
divergence into the loss function. Unlike traditional two-stage pruning pipelines, our approach
eliminates the need for post-hoc pruning or retraining.

The proposed method consistently outperforms both the base and L2-retrained models. It
achieves an average increase of 2% in weight sparsity and a 4–5×reduction in effective FLOPs
under activation pruning. Moreover, the framework generalizes well across both same-size and
large-to-small distillation settings, effectively promoting sparsity in diverse scenarios.
