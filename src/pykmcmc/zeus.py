"""
Usage: zeus [options] <data> <accmc>

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
    --weight-branch <branch>     Branch name for event weight [default: Weight]
    --silent                     Silent mode (disable output)
    -h --help                    Show this help message

"""
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import zeus
from docopt import docopt
from jax import random
from rich.console import Console

import pykmcmc.likelihood
import pykmcmc.reader


def run_zeus():
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

    ngen = args["--ngen"]
    if not ngen:
        ngen = accmc.N

    # Create our likelihood function
    likelihood = pykmcmc.likelihood.Likelihood(data, accmc, ngen)

    ndim = 23
    nwalkers = int(args["--walkers"])
    steps = int(args["--steps"])

    if args["--seed"]:
        seed = int(args["--seed"])
    else:
        seed = int(datetime.now().timestamp() * 1_000_000)
    key = random.PRNGKey(seed)
    p0 = random.ball(key, d=ndim, shape=(nwalkers,)) * 100
    sampler = zeus.EnsembleSampler(nwalkers, ndim, likelihood.log_likelihood)
    save_callback = zeus.callbacks.SaveProgressCallback(args["--output"], ncheck=10)
    # min_iter_callback = zeus.callbacks.MinIterCallback(nmin=500)
    sampler.run_mcmc(p0, steps, callbacks=[save_callback])
    samples = sampler.get_chain()
    plt.figure(figsize=(16, 4))
    plt.plot(samples[:, :, 0], alpha=0.5)
    plt.show()
