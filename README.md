# PyKMCMC

![Top Language](https://img.shields.io/github/languages/top/denehoffman/PyKMCMC) ![Total Lines](https://img.shields.io/tokei/lines/github/denehoffman/PyKMCMC) ![Code Size](https://img.shields.io/github/languages/code-size/denehoffman/PyKMCMC) ![Last Commit](https://img.shields.io/github/last-commit/denehoffman/PyKMCMC) ![License](https://img.shields.io/github/license/denehoffman/PyKMCMC)

PyKMCMC is a Python project designed to calculate the K-matrix amplitude for the $K_SK_S$ channel of the GlueX dataset and perform an MCMC analysis (Markov-Chain Monte Carlo).

## Table of Contents

- [Introduction](#introduction)
  - [The K-Matrix Amplitude](#the-k-matrix-amplitude)
  - [Constructing a Likelihood Function](#constructing-a-likelihood-function)
  - [Markov-Chain Monte Carlo](#markov-chain-monte-carlo)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Demo](#demo)
- [License](#license)
- [Tasks](#tasks)

## Introduction

The PyKMCMC project aims to provide a convenient and efficient way to calculate the K-matrix amplitude for the $K_SK_S$ channel of the GlueX dataset and perform an MCMC analysis. The K-matrix amplitude is an essential component in the study of hadronic scattering processes and is widely used in particle physics research.

### The K-Matrix Amplitude

The K-matrix parameterization[^1] is similar to a sum of Breit-Wigners, but it preserves unitarity for nearby resonances. The complex amplitude for the $i$-th final-state channel is written as:
```math
F_i(s;\vec{\beta}) = \sum_j \left( I + K(s) C(s)\right)_{ij}^{-1} \cdot P_j(s;\vec{\beta})
```
where $s$ is the invariant mass squared of $K_SK_S$ for a particular event, $I$ is the identity matrix, $\beta$ is a vector of complex couplings from the initial state to each resonance, and the other matrices and vectors are defined as follows.

First, we define the K-matrix as:
```math
K_{ij}(s) = \left(\frac{s - s_0}{s_\text{norm}}\right)\sum_{\alpha} B^{\ell}\left(q_i(s),q_i(m_\alpha^2)\right) \cdot \left(\frac{g_{\alpha,i}g_{\alpha,j}}{m_\alpha^2 - s} + c_{ij}\right) \cdot B^{\ell}\left(q_j(s),q_j(m_\alpha^2)\right)
```
Here we sum over resonances (labeled $\alpha$) which have a real mass $m_\alpha$ and real coupling $g_{\alpha,i}$ to the $i$-th channel. The term in front accounts for the Adler zero in chiral perturbation theory and must be included when handling the $f_0$ K-matrix, as one of the resonances is near the pion mass. Additionally, we can add real terms $c_{ij}$ and preserve unitarity. The outer functions are ratios of Blatt-Weisskopf barrier functions:
```math
B^{\ell}\left(q_i(s),q_i(m_\alpha^2)\right) = \frac{b^{\ell}\left(q_i(s)\right)}{b^{\ell}\left(q_i(m_\alpha^2)\right)}
```
where
```math
b^{\ell}(q) = \begin{cases} 1 & \ell = 0 \\ \sqrt{\frac{13z^2}{(z-3)^2 + 9z}},\ z = \frac{q^2}{q_0^2} & \ell = 2 \end{cases}
```
where $q_0$ is the effective centrifugal barrier momentum, set to $q_0 = 0.1973\text{ GeV}$ in the code. Currently, the barrier factors for $\ell\neq 0, 2$ are not implemented as they are not used in this channel's analysis.

The functions $q_i(s)$ correspond to the breakup momentum of a particle with invariant mass squared of $s$ in the $i$-th channel:
```math
q_i(s) = \rho_i(s)\frac{\sqrt{s}}{2}
```
```math
\rho_i(s) = \sqrt{\chi^+_i(s)\chi^-_i(s)}
```
```math
\chi^{\pm}_i(s) = 1 - \frac{(m_{i,1} \pm m_{i,2})^2}{s}
```
where $m_{i,1}$ and $m_{i,2}$ are the masses of the daughter particles in the $i$-th channel.

Next, $C(s)$ is the Chew-Mandelstam matrix. This is a diagonal matrix whose diagonal elements are given by the Chew-Mandelstam function[^2]:
```math
\begin{align}
C_{ii}(s) &= C_{ii}(s_{\text{thr}}) - \frac{s - s_{\text{thr}}}{\pi}\int_{s_{\text{thr}}}^{\infty} \text{d}s' \frac{\rho_i(s')}{(s'-s)(s'-s_{\text{thr}})} \\
          &= C(s_{\text{thr}}) + \frac{\rho_i(s)}{\pi}\ln\left[\frac{\chi^+_i(s)+\rho_i(s)}{\chi^+_i(s)-\rho_i(s)}\right] - \frac{\chi^+_i(s)}{\pi}\frac{m_{i,2}-m_{i,1}}{m_{i,1}+m_{i,2}}\ln\frac{m_{i,2}}{m_{i,1}}
\end{align}
```
with $s_{\text{thr}} = (m_{i,1}+m_{i,2})^2$. Additionally, we chose $C(s_{\text{thr}}) = 0$.

The final piece of the amplitude is the P-vector, which has a very similar form to the K-matrix:
```math
P_{j}(s;\vec{\beta}) = \sum_{\alpha} \left(\frac{\beta_{\alpha}g_{\alpha,j}}{m_\alpha^2 - s}\right) \cdot B^{\ell}\left(q_j(s),q_j(m_\alpha^2)\right)
```
where $\beta_\alpha$ is the complex coupling from the initial state to the resonance $\alpha$. In this analysis, the $\beta_\alpha$ factors are the free parameters in the fit. Note that you can add complex terms to the P-vector in much the same way as real terms can be added to the K-matrix without violating unitarity.

In the $K_SK_S$ channel, we have access to resonances with even total angular momentum and isospin $I=0,1$. We label these as $f$ and $a$ particles respectively, and at the energies GlueX accesses, we expect to see several $f_0$, $f_2$, $a_0$, and $a_2$ resonances. Since this project only analyzes one channel, all of the input parameters except for the values of each $\beta_\alpha$ are fixed in the fit according to published results by Kopf et. al[^1]. The spin-2 resonances are additionally multiplied by a factor of $Y_{J=2}^{M=2}\left(\theta_{\text{HX}},\phi_{\text{HX}}\right)$, a D-wave spherical harmonic with moment $M=2$ acting on the spherical angles in the helicity frame.

### Constructing a Likelihood Function

We can form an intensity density function by coherently summing the K-matrices for each type of particle:
```math
\mathcal{I}(s,\Omega;\vec{\beta}) = \lvert F^{f_0}_2(s;\vec{\beta}) + F^{a_0}_1(s;\vec{\beta}) + Y_2^2(\Omega)\left(F^{f_2}_2(s;\vec{\beta}) + F^{a_2}_2(s;\vec{\beta})\right) \rvert^2
```
This function is not normalized and does not account for the detector's acceptance/efficiency, the probability of an event getting detected and making it through all analysis actions. We can, of course, define the normalization as
```math
\mu = \int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega
```
where $\eta(s,\Omega)$ is the detector acceptance as a function of the observables in this analysis. This allows us to write a probability density function (PDF):
```math
\mathcal{P}(s,\Omega;\vec{\beta}) = \frac{1}{\mu}\mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega)
```
The extended maximum likelihood can be written as a product of these PDFs for each event times a Poissonian distribution:
```math
\mathcal{L}(\vec{\beta}) = \frac{e^{-\mu}\mu^N}{N!}\prod_i^N\mathcal{P}(s_i,\Omega_i;\vec{\beta})
```
Since the values for each evaluation of the PDF will be $\in(0,1)$, it is unstable to multiply them. Instead, we take the natural logarithm and minimize the negative log-likelihood:
```math
\begin{align}
\ln\mathcal{L}(\vec{\beta}) &= -\mu + N\ln(\mu) - \ln(N!) + \sum_i^N \ln(\mathcal{P}(s_i,\Omega_i;\vec{\beta})) \\
                            &= \sum_i^N \left[\ln\left(\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) + \ln\left(\eta(s_i,\Omega_i)\right) - \ln(\mu) \right] -\mu + N\ln(\mu) - \ln(N!) \\
                            &= \sum_i^N \left[\ln\left(\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) + \ln\left(\eta(s_i,\Omega_i)\right)\right] - N\ln(\mu) -\mu + N\ln(\mu) - \ln(N!) \\
                            &= \sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) -\mu - \ln(N!) + \sum_j^N \ln\eta(s_j,\Omega_j) \\
                            &= \sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) -\int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega - \ln(N!) + \sum_j^N \ln\eta(s_j,\Omega_j) \\
\end{align}
```
We must now compute this integral, but in general it does not have an analytic form, so we resort to Monte Carlo methods. Using the Mean Value Theorem, we know that integrating a function over a domain $D$ with area $A$ gives us the average value of that function times $A$:
```math
\int_D f(x)\text{d}x = A\langle f(x) \rangle
```
We can therefore use a Monte Carlo sample, letting $\eta(s,\Omega)$ be equal to $1$ for accepted events and $0$ for rejected events, to numerically compute the average:
```math
4\pi(s_{\text{max}} - s_{\text{min}})\langle \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \rangle \approx \frac{4\pi(s_{\text{max}} - s_{\text{min}})}{N_{\text{gen}}}\sum_{i}^{N_{\text{acc}}}\mathcal{I}(s_i,\Omega_i;\vec{\beta}) = \int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega
```
All together, we end up with
```math
-\ln\mathcal{L}(\vec{\beta}) = -\left(\sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) - \frac{4\pi(s_{\text{max}} - s_{\text{min}})}{N_{\text{gen}}}\sum_{i}^{N_{\text{acc}}}\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) + \ln(N!) - \sum_j^N \ln\eta(s_j,\Omega_j)
```
We can, of course, compute $\ln(N!)$, but the final term is still unknown. However, it doesn't depend on the free parameters $\vec{\beta}$, so it vanishes when we minimize with respect to $\vec{\beta}$ (as does $\ln(N!)$, but it's inexpensive to calculate and we can do so if we want to).

### Markov-Chain Monte Carlo

[TODO]

## Installation

The easiest way to install PyKMCMC is by first cloning the GitHub repository and then installing through `pip`:
```shell
git clone git@github.com:denehoffman/PyKMCMC.git
cd PyKMCMC
pip install .
```

## Usage

PyKMCMC includes two methods for running and analyzing MCMC chains. The first is `kmcmc`:
```shell
kmcmc --help
Usage:
    kmcmc [options] <data> <accmc>

Options:
    -o --output <output>         Output file name ending with '.h5' [default: chain.h5]
    -f --force                   Force overwrite if the output file already exists
    -s --steps <steps>           Number of steps [default: 500]
    -w --walkers <walkers>       Number of walkers [default: 70]
    --ngen <ngen>                Size of generated Monte Carlo (defaults to accepted size)
    --seed <seed>                Seed value for random number generator
    --mass-branch <branch>       Branch name for mass [default: M_FinalState]
    --cos-theta-branch <branch>  Branch name for cos theta [default: HX_CosTheta]
    --phi-branch <branch>        Branch name for phi [default: HX_Phi]
    --ncores <cores>             Number of cores [default: 1]
    --mpi                        Enable MPI
    --silent                     Silent mode (disable output)
    -h --help                    Show this help message

Notes:
    - The program will exit if either <data> or <accmc> file does not exist or does
      not end with '.root'.
    - The program will exit if the output file already exists, and the -f, --force
      option is not provided.
    - If the <seed> is not specified, it will default to a random integer determined
      by the system time.

Examples:
    kmcmc data.root accmc.root -o mychain.h5 -s 100
    - Runs 100 steps

    mpiexec -n 4 kmcmc data.root accmc.root --mpi
    - Run using OpenMPI on 4 cores

    kmcmc data.root accmc.root --ncores 4
    - Run using Python multiprocessing on 4 cores
```

The other method is `kplot`, which is still under development, although a preliminary version is included in the current package.

## Data Requirements

At a bare minimum, `kmcmc` requires two files to run, a data file and a Monte Carlo file. The both files must have four branches containing the mass, event weight, and two spherical angles related to the decay.

## Demo

[TODO]

## License

This project is licensed under the [MIT License](LICENSE).

## Tasks

- [ ] Finish documentation
- [ ] Finish unit tests
- [ ] Finish plotting


[^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $\bar{p}p$-, $\pi^-p$- and $\pi\pi$-Data. *Eur. Phys. J. C* **81**, 1056 (2021). [https://doi.org/10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
[^2]: Wilson, D. J., Dudek, J. J., Edwards, R. G. & Thomas, C. E. Resonances in coupled $\pi K$, $\eta K$ scattering from lattice QCD. *Phys. Rev. D* **91**, 054008 (2015). [https://doi.org/10.1103/PhysRevD.91.054008](https://doi.org/10.1103/PhysRevD.91.054008)
