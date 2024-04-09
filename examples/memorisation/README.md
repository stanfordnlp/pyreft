# Model Interpretability with ReFT

Based on the notebook [`reft_power.ipynb`](https://github.com/stanfordnlp/pyreft/blob/main/examples/memorisation/reft_power.ipynb).

The goal is to show some ways to explore why ReFT works. This direction focuses specifically on model memorisation. There will be other directions for model interpretability enabled by ReFT which is left for your to explore.

## Memorisation

The motivation for this is simple: **to see how many words can a single neuron intervention store in a linear subspace**. The minimum setup is to train an intervention to recover a long sequence. Please see the notebook for details.
