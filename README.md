# ConditionalDCGAN.
A PyTorch implementation of Conditional DCGAN.

## Related Papers

Alec Radford, Luke Metz and Soumith Chintala:
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.
https://arxiv.org/abs/1511.06434

Mehdi Mirza and Simon Osindero: Conditional Generative Adversarial Nets. 
https://arxiv.org/pdf/1411.1784.pdf

## Training

1. Clone this repository and move to the directory.

```shell
% clone https://github.com/mps-research/ConditionalDCGAN.git
% cd ConditionalDCGAN
```

2. Download "img_align_celeba.zip" from [CelebA Dataset web page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

3. Put the zip into the data directory and unzip it.

```shell
% cd data
% unzip img_align_celeba.zip
```

4. Move to the repository root directory and build "cdcgan" docker image and run the image inside of a container.

```shell
% cd ..
% docker build -t cdcgan .
% ./train.sh
```

## Checking Training Results

```shell
% ./run_tensorboard.sh
```