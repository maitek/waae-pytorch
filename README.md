# Wasserstein Adversarial Autoencoder Pytorch

This is a Pytorch implementation of an Adversarial Autoencoder (https://arxiv.org/abs/1511.05644) using Wasserstein loss (https://arxiv.org/abs/1701.07875) on the discriminator. The Wasserstein loss allows for more stable training than the Vanilla GAN loss proposed in the original paper.

The Encoder and Decoder uses an architecture similar to DCGAN (https://arxiv.org/abs/1511.06434)

## Reconstructed:
![alt text](https://raw.githubusercontent.com/maitek/waae-pytorch/master/results/reconstruction.png)

## Generated images by sampling from embedding:
![alt text](https://raw.githubusercontent.com/maitek/waae-pytorch/master/results/generated.png)

Special thanks to wiseodd for his educational generative model repository:
https://github.com/wiseodd/generative-models
