# Thompson Sampling via Local Uncertainty

This repo corresponds to the *[Thompson Sampling via Local Uncertainty](https://arxiv.org/pdf/1910.13673.pdf)* paper, published in
[ICML](https://icml.cc/) 2020. We propose a new probabilistic modeling framework for Thompson sampling, where local latent variable uncertainty is used to sample the mean reward. We implemented our algorithms LU-Gauss/SIVI based on the [Benchmark](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits) proposed by [Carlos Riquelme](http://rikel.me). 

Please, use the following when citing the code or the paper:

```
@article{zhendong2020luts, 
title={Thompson Sampling via Local Uncertainty},
author={Wang, Zhendong and Zhou, Mingyuan},
journal={International Conference on Machine Learning, ICML.}, 
year={2020}}
```

## Dependencies

This code is based on Python 2.7, with the main dependencies being TensorFlow==1.14 and other dependencies stated in the [Benchmark](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits)

## Dataset

You can download the required dataset from the benckmark. 

## Training

You can run comparison of all algorithms from *run_bandit.py* file, with one argument for a specific dataset. It will save the output results in one *.npz* file. 

```
    python run_bandit.py mushroom
```

## Acknowledgement

We thank greatly Riquelme et al. for making their code public.
