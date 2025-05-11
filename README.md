# Diffusion vs. GANs

This repository compares **Denoising Diffusion Probabilistic Models (DDPMs)** and **Generative Adversarial Networks (GANs)** on the FashionMNIST dataset. Both methods aim to generate realistic images from noise, but they do so in fundamentally different ways.

**Note**: All implementations and analyses are located in notebooks/DDPM-vs-GANs.ipynb.

## 1. Mathematical Overview

### 1.1 GAN Objective

A standard GAN involves two players: a **Generator** $G$ and a **Discriminator** $D$. The goal is to solve:\

```math
\min_{\theta} \max_{\phi} V(G_\theta, D_\phi) = \mathbb{E}_{\mathbf{x} \sim \textbf{p}_{\textrm{data}}}[\log D_\phi(\textbf{x})] + 
\mathbb{E}_{\mathbf{z} \sim p(\textbf{z})}[\log (1-D_\phi(G_\theta(\textbf{z})))]
```
- **$\mathbf{x}$**: A real image sampled from the dataset, i.e., $\mathbf{x} \sim p_{\text{data}}$.
- **$\mathbf{z}$**: Random noise sampled from a prior distribution, i.e., $\mathbf{z} \sim p(\mathbf{z})$.
- **$G_{\theta}(\mathbf{z})$**: The generated (fake) image produced by the generator $G_{\theta}$.
- **$D_{\phi}(\mathbf{x})$**: The probability output by the discriminator that $\mathbf{x}$ is a real image.
- **$D_{\phi}(G_{\theta}(\mathbf{z}))$**: The probability output by the discriminator that the generated image $G_{\theta}(\mathbf{z})$ is real.

### 1.2 DDPM Objective

A DDPM gradually **diffuses** data by adding noise step-by-step and then **denoises** in reverse. Training is done by matching predicted noise $\epsilon_\theta(\mathbf{x}_t, t)$ to the actual noise $\epsilon$ at each step $t$:
```math
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\Big[
\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t)\|^2
\Big].
```

During sampling, we iteratively remove noise from an initial Gaussian sample $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to recover $\mathbf{x}_0$.

## 2. Implemented Models

1. **GAN**  
2. **Conditional GAN (cGAN)**
3. **DDPM**  
4. **Conditional DDPM**

All models use **FashionMNIST** as the training dataset, resized to $32 \times 32$ and normalized to $[-1,1]$.



## 3. Results

### 3.1 Sample Images

#### GAN Samples
![gan_samples.png](results/gan_samples.png)
#### DDPM Samples
![ddpm_samples.png](results/ddpm_samples.png)
#### Conditional GAN
![cgan_samples.png](results/cgan_samples.png)
#### Conditional DDPM Samples
![cddpm_samples.png](results/cddpm_samples.png)

### 3.2 Loss Curves


![Loss Curves](results/loss_curve.png)



## 4. Key Takeaways

1. **Stability vs. Detail**  
   - GAN training is adversarial and can be unstable but sometimes produces sharper details.
   - DDPM training is more stable and consistently generates coherent images.

2. **Mode Collapse**  
   - GANs can suffer from mode collapse, repeatedly generating similar outputs.
   - DDPMs do not typically exhibit mode collapse.

3. **Conditioning**  
   - Both models can be conditioned on class labels to generate class-specific images (cGAN, Conditional DDPM).

## References
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239

- Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. NeurIPS 2014

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747
