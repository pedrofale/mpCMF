# pCMF

We first discuss the original pCMF on small data sets. Then, we apply it to large data sets by making use of Stochastic Variational Inference. We assess the performance on these data sets and, if it is degraded, we check if the model assumptions don't hold for large data sets. By sampling from the model and comparing with the observations, we can criticize the model assumptions and make new ones, which lead to a different model.

We also develop HpCMF, which places priors on the hyperparameters of pCMF to force sparsity and make zero-inflation more expressive.
