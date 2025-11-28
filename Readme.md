# Data-Free Model-Related Attacks Unleashing the Potential of Generative AI
## About
This is the source code that uses large models to steal the data distribution of (small) victim models.
We implement our attacks on five tasks: MNIST, CIFAR10, SKIN_CANCER, IMDB, Private PET.<br>

## Attack Intuition
We compared the distribution scatter plot of the raw training data with that of the generated data. Taking CIFAR10 task as an example, the figure is as follow. It is our attack intuition since the distributions of both are similar.
<p align="center">
    <img src="./generated_demo/compare.png" alt="compare" width="1000" height="420">
</p>

## Description
Each task folder includes a more detailed Readme.md, which outlines the implementation steps.

If you have any questions, please do not hesitate to contact me.