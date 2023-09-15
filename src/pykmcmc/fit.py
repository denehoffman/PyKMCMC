"""
Usage: fit [options] <data> <accmc>

Options:
    -o --output <output>         Output file name ending with '.h5' [default: chain.h5]
    -f --force                   Force overwrite if the output file already exists
    --ngen <ngen>                Size of generated Monte Carlo (defaults to accepted size)
    --seed <seed>                Seed value for random number generator
    --mass-branch <branch>       Branch name for mass [default: M_FinalState]
    --cos-theta-branch <branch>  Branch name for cos theta [default: HX_CosTheta]
    --phi-branch <branch>        Branch name for phi [default: HX_Phi]
    --weight-branch <branch>     Branch name for event weight [default: Weight]
    --silent                     Silent mode (disable output)
    -h --help                    Show this help message

"""
import sys
from datetime import datetime
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from iminuit import Minuit
from jax import random
from rich.console import Console

import pykmcmc.likelihood
import pykmcmc.reader


def run_fit():
    args = docopt(__doc__)
    console = Console(quiet=args["--silent"], record=True)
    error_console = Console(stderr=True, style="bold red")
    data_path = Path(args["<data>"])
    if not data_path.exists():
        error_console.print(f"File not found: {data_path}")
        sys.exit(1)
    if not data_path.suffix == ".root":
        error_console.print(f"Data path must be a .root file ({data_path})")
        sys.exit(1)
    accmc_path = Path(args["<accmc>"])
    if not accmc_path.exists():
        error_console.print(f"File not found: {accmc_path}")
        sys.exit(1)
    if not accmc_path.suffix == ".root":
        error_console.print(f"Monte Carlo path must be a .root file ({accmc_path})")
        sys.exit(1)
    output_path = Path(args["--output"])
    if output_path.exists() and not args["--force"]:
        error_console.print(
            f"The file {output_path} already exists, use --force to override"
        )
        sys.exit(1)
    with console.status("Loading DATA..."):
        data = pykmcmc.reader.read_file(
            data_path,
            args["--mass-branch"],
            args["--cos-theta-branch"],
            args["--phi-branch"],
            args["--weight-branch"],
        )
        console.print("Loaded DATA")
    with console.status("Loading ACCMC..."):
        accmc = pykmcmc.reader.read_file(
            accmc_path,
            args["--mass-branch"],
            args["--cos-theta-branch"],
            args["--phi-branch"],
            args["--weight-branch"],
        )
        console.print("Loaded ACCMC")

    if args["--ngen"]:
        ngen = int(args["--ngen"])
    else:
        ngen = accmc.N

    # Create our likelihood function
    likelihood = pykmcmc.likelihood.Likelihood(data, accmc, ngen)
    init = np.random.rand(23) * 300
    grad_nll = jax.grad(likelihood.negative_log_likelihood)
    nll = likelihood.negative_log_likelihood
    with console.status("Compiling NLL..."):
        val = nll(init)
        console.print(f"Done! NLL(init) = {val}")
    with console.status("Compiling grad(NLL)..."):
        val = grad_nll(init)
        console.print(f"Done! ∇NLL(init) = {val}")
    m = Minuit(nll, init, grad=grad_nll)
    m.errordef = Minuit.LIKELIHOOD
    m.strategy = 0
    m.migrad()
    print(m)
    best_pars = np.array(m.values)
    fit_weights = accmc.calc_weights(best_pars)
    fw_sum = np.sum(fit_weights)
    dw_sum = np.sum(data.weight)
    bins = 70
    f0_weights = accmc.calc_weights_f0(best_pars)
    f2_weights = accmc.calc_weights_f2(best_pars)
    a0_weights = accmc.calc_weights_a0(best_pars)
    a2_weights = accmc.calc_weights_a2(best_pars)
    plt.hist(data.mass, weights=data.weight, bins=bins, histtype="step", label="Data")
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
    plt.savefig("bestplot_minuit.svg")
