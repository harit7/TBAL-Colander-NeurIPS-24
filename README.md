# Pearls from Pebbles: Improved Confidence Functions for Auto-labeling 

![alt text](figs/auto-labeling-sketch.png)

**TL;DR:** In this work we analyze optimal confidence function for threshold-based autolabeling (TBAL)

> **Keywords:** Auto Labeling, Confidence Functions, Active Learning, Selective Classification

# Instructions to run code
First things first, lets create the conda environment as follows,

## Environment 

We recommend you create a conda environment as follows.

```
conda env create -f environment.yml
```

and activate it with

```
conda activate tbal
```

Now lets run some examples,

## Sample Usage
To get started we recommend heading to the sample scripts directory and running the following:

1. `cd ./scripts`
2. For the MNIST LeNet setup we used in our experiments. Run `./run_mnist_tbal_eval_full_fixed.sh` in your command line. This script executes the cross-product of hyperparameter-tuned configurations from TBAL, train-time methods, and post-hoc methods, as seen in Table 1 of our experiments. The configurations are located in the `./configs/calib-exp/hyp-search/tbal/` directory. The configurations used for MNIST LeNet are located in the files:
    - TBAL configurations: `mnist_lenet/mnist_lenet_common_fixed.json`
    - Train-time method configurations: `mnist_lenet/mnist_lenet_train_fixed.json`
    - Post-hoc methods configurations: `mnist_lenet/mnist_lenet_post_fixed_std_xent.json`.
(Similarly, other scripts to run CIFAR-10-CNN, TinyImagenet-MLP, 20 Newsgroups-MLP are located in the same directory.)

## Compute used in our experiments 
**GPUs**: NVIDIA RTX A6000, NVIDIA GeForce RTX 4090
