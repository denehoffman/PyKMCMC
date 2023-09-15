"""
Usage:
    zplot [options] chain <chains.h5>
    zplot [options] corner <chains.h5>
    zplot [options] best <chains.h5> <data.root> <accmc.root>

Options:
    --thin <n>                   Use every nth value [default: 1]
    --burn <n>                   Burn the first n steps [default: 0]
    --mass-branch <branch>       Branch name for mass [default: M_FinalState]
    --cos-theta-branch <branch>  Branch name for cos theta [default: HX_CosTheta]
    --phi-branch <branch>        Branch name for phi [default: HX_Phi]
    --weight-branch <branch>     Branch name for event weight [default: Weight]
"""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import zeus
from corner import corner
from docopt import docopt

import pykmcmc.reader


def zplot():
    args = docopt(__doc__)
    thin = int(args["--thin"])
    burn = int(args["--burn"])
    ndim = 23

    with h5py.File(args["<chains.h5>"], "r") as hf:
        samples = np.copy(hf["samples"])
        logprob_samples = np.copy(hf["logprob"])
        burned_samples = samples[burn::thin]
        burned_logprob_samples = logprob_samples[burn::thin]
    burned_flat_samples = np.reshape(
        burned_samples,
        (burned_samples.shape[0] * burned_samples.shape[1], burned_samples.shape[2]),
    )
    burned_flat_logprob_samples = np.reshape(
        burned_logprob_samples,
        burned_logprob_samples.shape[0] * burned_logprob_samples.shape[1],
    )
    if args["chain"]:
        plt.figure(figsize=(16, 4 * ndim))
        for n in range(ndim):
            plt.subplot2grid((ndim, 1), (n, 0))
            plt.plot(burned_samples[:, :, n], alpha=0.5)
        plt.tight_layout()
        plt.savefig("chainplot.svg")
    elif args["corner"]:
        # fig, axes = zeus.cornerplot(burned_flat_samples)
        fig = corner(burned_flat_samples)
        plt.savefig("cornerplot.svg")
    elif args["best"]:
        data_path = Path(args["<data.root>"])
        accmc_path = Path(args["<accmc.root>"])
        data = pykmcmc.reader.read_file(
            data_path,
            args["--mass-branch"],
            args["--cos-theta-branch"],
            args["--phi-branch"],
            args["--weight-branch"],
        )
        accmc = pykmcmc.reader.read_file(
            accmc_path,
            args["--mass-branch"],
            args["--cos-theta-branch"],
            args["--phi-branch"],
            args["--weight-branch"],
        )
        i_best = np.argmax(burned_flat_logprob_samples)
        best_pars = burned_flat_samples[i_best]
        fit_weights = accmc.calc_weights(best_pars)
        fw_sum = np.sum(fit_weights)
        dw_sum = np.sum(data.weight)
        bins = 70
        f0_weights = accmc.calc_weights_f0(best_pars)
        f2_weights = accmc.calc_weights_f2(best_pars)
        a0_weights = accmc.calc_weights_a0(best_pars)
        a2_weights = accmc.calc_weights_a2(best_pars)
        plt.hist(
            data.mass, weights=data.weight, bins=bins, histtype="step", label="Data"
        )
        plt.hist(
            accmc.mass,
            weights=fit_weights / fw_sum * dw_sum,
            bins=bins,
            histtype="step",
            label="Fit",
        )
        plt.hist(
            accmc.mass,
            weights=f0_weights / fw_sum * dw_sum,
            bins=bins,
            histtype="step",
            label="Fit f0",
        )
        plt.hist(
            accmc.mass,
            weights=f2_weights / fw_sum * dw_sum,
            bins=bins,
            histtype="step",
            label="Fit f2",
        )
        plt.hist(
            accmc.mass,
            weights=a0_weights / fw_sum * dw_sum,
            bins=bins,
            histtype="step",
            label="Fit a0",
        )
        plt.hist(
            accmc.mass,
            weights=a2_weights / fw_sum * dw_sum,
            bins=bins,
            histtype="step",
            label="Fit a2",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("bestplot.svg")
