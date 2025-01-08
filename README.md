# Bayesian Finite Mixture Modelling

MATLAB code for Bayesian Computation of Finite Mixture Models.

## SUMMARY

### Treatment Response Models (Finite Mixture of 2SLS Models)

#### Continuous Treatment

$`
\begin{aligned}
D_i &= \gamma_{0,\tilde{c}_i} + \mathbf{Z}_i\mathbf{\gamma}_{z,\tilde{c}_i} + v_i\\
Y_i &=  \beta_{0,\tilde{c}_i} + \mathbf{X}_i\mathbf{\beta}_{x,\tilde{c}_i} + \delta_{\tilde{c}_i} D_i + \epsilon_i\\
\end{aligned}
`$

where

$`
\begin{aligned} 
\begin{bmatrix}
v_i \\ u_i 
\end{bmatrix} \bigm\vert \mathbf{Z},\mathbf{X} &\overset{ind}{\sim} \mathcal{N}\left(
\begin{bmatrix}
0 \\ 0
\end{bmatrix}, 
\begin{bmatrix}
\sigma_{v, \tilde{c}_i}^2 & \sigma_{vu,\tilde{c}_i}\\
\sigma_{vu, \tilde{c}_i} & \sigma_{u,\tilde{c}_i}^2\\
\end{bmatrix}
\right)
\end{aligned} \equiv \mathcal{N}\left( \mathbf{0}, \mathbf{\Sigma}_{\tilde{c}_i} \right)
`$

and $`\text{Pr}(\tilde{c}_i = g) = \pi_g, \quad \text{for } g = 1,\ldots,G \text{ and } \sum_{g=1}^G \pi_g = 1`$.

#### Binary Treatment

$`
\begin{aligned}
D^*_i &= \gamma_{0,\tilde{c}_i} + \mathbf{Z}_i\mathbf{\gamma}_{z,\tilde{c}_i} + v_i\\
D_i &= \mathbb{1}\{D^*_i \geq 0\}\\
Y_i &=  \beta_{0,\tilde{c}_i} + \mathbf{X}_i\mathbf{\beta}_{x,\tilde{c}_i} + \tau_{\tilde{c}_i} D_i + \epsilon_i\\
\end{aligned}
`$

where

$`
\begin{aligned} 
\begin{bmatrix}
v_i \\ u_i 
\end{bmatrix} \bigm\vert \mathbf{Z},\mathbf{X} &\overset{ind}{\sim} \mathcal{N}\left(
\begin{bmatrix}
0 \\ 0
\end{bmatrix}, 
\begin{bmatrix}
\sigma_{v, \tilde{c}_i}^2 & \sigma_{vu,\tilde{c}_i}\\
\sigma_{vu, \tilde{c}_i} & \sigma_{u,\tilde{c}_i}^2\\
\end{bmatrix}
\right)
\end{aligned} \equiv \mathcal{N}\left( \mathbf{0}, \mathbf{\Sigma}_{\tilde{c}_i} \right)
`$

and $`\text{Pr}(\tilde{c}_i = g) = \pi_g, \quad \text{for } g = 1,\ldots,G \text{ and } \sum_{g=1}^G \pi_g = 1`$.

## USAGE

+ Gibbs samplers: [`MCMCsamplers`](https://github.com/duongtrinhss/Bayesian-Finite-Mixture-Modelling/tree/main/MCMCsamplers)

+ Permutation algorithms to handle label-switching: [`permutation`](https://github.com/duongtrinhss/Bayesian-Finite-Mixture-Modelling/tree/main/permutation)

+ Simulations: [`SyntheticMonteCarlo`](https://github.com/duongtrinhss/Bayesian-Finite-Mixture-Modelling/tree/main/SyntheticMonteCarlo)


