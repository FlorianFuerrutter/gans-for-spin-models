# gans-for-spin-models 

[Thesis](BaThesis_Fuerrutter_Florian.pdf)

## Abstract
Machine learning has experienced an uninterrupted growth over the last few years. Advancements in research and computation hardware allow the application of neural networks in a diverse spectrum. Motivated by remarkable results in generating indistinguishable fake images of humans and landscapes (e.g., Deep-fakes) physical problems are getting more attention. In this thesis, we look at generative adversarial networks (GAN) which are able to learn the hidden distribution of a training set sampled from a physical system. Based on conditional GANs we show the Ising model can be learned and reproduced at unseen thermodynamic control parameters. We propose the SpinGAN architecture that catches the phase transition of the 2D Ising model with great accuracy. The deterministic neural network allows us to evolve single distinct Ising states through the temperature range. Utilizing the exact differentiation of the network we present a new GAN fidelity to identify critical temperatures without knowledge of the physical system.

## Ising model
Advancements in *quantum theory*, especially the inference of the spin concept, made a basic understanding of para- and ferromagnetism in solid matter possible. In 1924 a statistical physics model of ferromagnetism was proposed by Ernst Ising, namely the broadly known Ising model. The mathematical model describes spins at discrete lattice sides that can be considered either classically or quantum theoretically. We restrict ourselves here to the classical Ising model where the spin at lattice point $i$ is discrete. The general Hamiltonian is given by:

$$H = -\sum_{\left\langle i,j \right\rangle} J_{i, j}  \sigma_i\sigma_j -\sum_{i} h_i \sigma_i$$

Here $\left\langle i,j \right\rangle$ means the sum over all adjacent lattice sides and $J_{i, j}$ is the exchange interaction strength between spin $i$ and $j$. The influence of an external field is denoted as $h_i$. The discrete spins are either up or down $\sigma_i = \pm1$.  In this thesis we focus on the two-dimensional case with a global interaction strength $J=J_{i, j}>0$ and no external field $h_i=0$.


![](/img/spin_lattice.png)

---
## Conditional GAN

![](/img/GAN_cond_concept.png)

![](/img/gan_goal.PNG)

---
## SpinGAN
For $L=64$.

**Discriminator**

![](/img/spinGAN_discriminator.png)

**Generator**

![](/img/spinGAN_generator.png)


---
## Model performance
Comparison of the GAN performance.


![](/img/gan_perf_m_e.png)

![](/img/gan_perf_chi_xi.png)


**SpinGAN: Magnetization $m$ histogram**

![](/img/gan_hist_m.png)

**SpinGAN: Energy $E$ histogram**

![](/img/gan_hist_e.png)


---
## State evolution
Fix the latent vector $z\in\mathcal{Z}$ and change the conditional temperature $T$.  Evolve a state from the ferromagnetic to the the paramagnetic regime.

![](/img/spin_gan_sample_large.png)

**Latent vector 1**

![](/img/state_evolution1.gif)

**Latent vector 2**

![](/img/state_evolution2.gif)

---
#### Table of contents
- [data folder](data)
- [train data](data/train)
- [simulation](simulation)
- [plotting](plotting)
