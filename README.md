# gans-for-spin-models
Bachelor thesis

![spin_lattice](/img/spin_lattice.png)


---
## Evaluation
Comparison of the GAN performance.

**Observables**
![](/img/gan_perf_m_e.png)
![](/img/gan_perf_chi_xi.png)


**SpinGAN: Magnetization $m$ histogram**
![](/img/gan_hist_m.png)

**SpinGAN: Energy $E$ histogram**
![](/img/gan_hist_e.png)



---
## State evolution
Fix the latent vector $z\in\mathcal{Z}$ and change the conditional temperature $T$.  Evolve a state from the ferromagnetic to the the paramagnetic regime.

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
