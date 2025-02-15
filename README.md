# Diffusion-Generated Image Detection

## Introduction
This project focuses on detecting diffusion-generated images using a robust dual loss learning approach. Inspired by recent research (DRCT), I applied a margin-based contrastive loss to perform contrastive learning between original and reconstructed images. This method helps the model effectively classify difficult samples, contributing to the development of a more generalized detector.

## Dual Loss Learning
Figure 3 illustrates the overall process of dual loss learning. Real and fake images were reconstructed using a stable diffusion inpainting model. I then extracted 1024-dimensional embeddings from the layer before the fully connected (FC) layer of ConvNeXt-base. Contrastive loss and reconstruction loss were calculated from these embeddings. The dual loss function is defined as follows:

```math
L_{dual} = L_{contrastive} + \lambda L_{recon}
```

where $\lambda$ is the weight applied to the reconstruction loss, set to $0.05$. Dual loss is combined with binary cross-entropy loss to enhance the ability of ConvNeXt to distinguish between fake and real images.

## Contrastive Loss
The contrastive loss follows the equation:

```math
L_{contrastive} = \frac{1}{N_{pos}} \sum_{(i,j) \in P} \left(1 - \text{dist}(z_i, z_j)\right)^2 + \frac{1}{N_{neg}} \sum_{(i,j) \in N} \max \left( \text{dist}(z_i, z_j) - m, 0 \right)^2
```

where $N_{pos}$ is the number of positive pairs, $P$ represents the positive pairs set, $N_{neg}$ is the number of negative pairs, and $N$ represents the negative pairs set. The distance metric $\text{dist}$ defaults to cosine distance.

Positive pairs consist of real reconstructed images, fake images, and fake reconstructed images, while negative pairs only include real images. As a result, the model learns to distinguish real reconstructed images more effectively from real images, leading to higher accuracy.

## Reconstruction Loss
The reconstruction loss is designed to increase the distance between real images and their reconstructions while reducing the distance between fake images and their reconstructions. It serves two main purposes:
1. Enhancing the training process efficiency.
2. Regulating the distance between real reconstructed images and fake reconstructed images.

By complementing contrastive loss, reconstruction loss accelerates training. Without it, training required over $15$ epochs, whereas adding reconstruction loss reduced the necessary epochs to fewer than $5$. Furthermore, if real reconstructed images and fake images become too similar, the model may overfit to the training data, degrading performance on images generated by different models. Since diffusion models differ in their generated image distributions, maintaining an appropriate distance via reconstruction loss improves generalization performance.

The reconstruction loss is defined as:

```math
L_{recon} = \frac{1}{N} \sum_i \max \left( \| x_i - \hat{x}_i \|_2^2 - \| y_i - \hat{y}_i \|_2^2 + \gamma, 0 \right)
```

where $N$ is the number of samples, and all image embeddings are normalized. $\gamma$ ensures a sufficient difference between real and fake reconstruction errors.

<p align="center"><img src="figures/Figure4.svg" alt="Pipeline" width="80%" /></p>

The figure above compares embedding distances before and after applying reconstruction loss. In (a), where only contrastive loss was used, the embeddings of three image types (real reconstructed images, fake images, and fake reconstructed images) are closely positioned. However, after adding reconstruction loss, as shown in (b), real reconstructed images maintain a slight distance from the other two types. The accuracy on the GenImage dataset increased from $77.18\%$ (contrastive loss only) to $81.01\%$, demonstrating the significance of reconstruction loss.

## Conclusion
This project implements a dual loss learning approach to improve the detection of diffusion-generated images. By integrating contrastive loss with reconstruction loss, the model effectively generalizes to various generative models. The approach not only improves accuracy but also accelerates training, making it practical for real-world applications.

This code is based on [DRCT](https://github.com/beibuwandeluori/DRCT.git).


