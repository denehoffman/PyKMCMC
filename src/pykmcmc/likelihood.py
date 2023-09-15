import jax.numpy as jnp

import pykmcmc.amplitude as amp


class Likelihood:
    def __init__(self, data, accmc, n_gen):
        self.data = data
        self.accmc = accmc
        self.n_gen = n_gen
        self.log_n_factorial = (
            self.data.N * jnp.log(self.data.N) - self.data.N
        )  # Stirling's approx

    def negative_log_likelihood(self, pars):
        cx_beta_r = amp.Amplitude.pars_to_cx_beta_r(*pars)
        data_sum = self.data.calc_log(cx_beta_r)
        accmc_sum = self.accmc.calc(cx_beta_r)
        return (
            -(data_sum - (4 * jnp.pi * (2.0**2 - 1.0**2) / self.n_gen) * accmc_sum)
            + self.log_n_factorial
        )

    def log_likelihood(self, pars):
        cx_beta_r = amp.Amplitude.pars_to_cx_beta_r(*pars)
        data_sum = self.data.calc_log(cx_beta_r)
        accmc_sum = self.accmc.calc(cx_beta_r)
        return (
            data_sum - (4 * jnp.pi * (2.0**2 - 1.0**2) / self.n_gen) * accmc_sum
        ) - self.log_n_factorial
