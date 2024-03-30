# Composable ReFT
Based on the notebook [`compreft.ipynb`](http://localhost:10008/notebooks/dev_tmp/reft/examples/composition/compreft.ipynb).


## Are ReFTs composable?

I have:

- **A ReFT for continuing sentences in German**
- **A ReFT for following instructions**

Can I just combine them and have an "instruction-following model that speaks German"? Let's see!

First of all, you need to know the notations of **subspace**, **linear subspace**, and **orthonormal linear subspaces**! You can read more about these in Atticus's [causal abstraction paper](https://arxiv.org/abs/2301.04709). Briefly, here is what they are:

- **subspace**: you can think of it as a single dimension of an NN's representation in the NN's original basis (learned one).
- **linear subspace**: representation in a changed basis, and the new basis is a linear combination (i.e., any rotation) of the original basis.
- **orthonormal linear subspaces**: if the new linear subspace is produced by an orthonormal projection, then each dimension (or sub-subspace, sorry about the confusion here) in that new basis is orthogonal to each other. Or more strictly speaking, *it maintains the orthogonality if the original basis has it*.

So for ReFT, we can theoretically leverage the notation of subspace, and train different subspaces for different tasks separately, and snap them together at the inference time! Let's see if it will work in practice.
