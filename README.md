# C65_project_proximal_algorithm

<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8.8-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.5.0-orange?logo=tensorflow">
</p>

## Introduction

This package is for numerical experiments in the report of C6.5. The code is based on the project **[StochasticProximalMethods](https://github.com/unc-optimization/StochasticProximalMethods)**. Special thanks for their code!!!!!!!!

The original paper for original code can be found in 

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh. **[Proxsarah: An efficient algorithmic framework for stochastic composite non-convex optimization](http://jmlr.org/papers/v21/19-248.html)**. <em>Journal of Machine Learning Research</em>, 21(110):1–48,2020.

## How to use

```text
python experiment_1.py -d fashion_mnist -a 1234 -so 24 -b 245 -ne 150 -r 1 -p 1
```
```text
python experiment_2.py -d cifar -a 123 -so 24 -b 245 -ne 700 -r 0 -p 1
```


## Changelist

There are several changes compared with original projects, which will be listed below:

* The name of original license file has changed to ``LICENSE_original``, please carefully check before using this project.
* Since the original project are designed to work on Tensorflow 1.\*.\*, all files are edited to be able to work on Tensorflow 2.\*.\* (by forcing using `tensorflow.compact.v1`).
* An parameter using to control the random noise has been added to ``argParser_edited.py``.
* Leaky ReLU method has been added in ``utils.py``.

## Added list

* Based on ``method_ProxSGD.py``, a vanilla SGD method has been added in ``method_SGD.py``.
* Based on their model, in ``model_experiment.py`` we have set two different DNNs for experiments.
* Details for experiments are in ``experiment_1.py`` and ``experiment_2.py``.