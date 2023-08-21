"""
Usage:
    kopt [options] <data> <accmc>

Options:
    -f --force                   Force overwrite if the output file already exists
    -s --steps <steps>           Number of steps [default: 500]
    -w --walkers <walkers>       Number of walkers [default: 70]
    -c --config <config>         TOML configuration file to set moves
    --ngen <ngen>                Size of generated Monte Carlo (defaults to accepted size)
    --seed <seed>                Seed value for random number generator
    --mass-branch <branch>       Branch name for mass [default: M_FinalState]
    --cos-theta-branch <branch>  Branch name for cos theta [default: HX_CosTheta]
    --phi-branch <branch>        Branch name for phi [default: HX_Phi]
    --weight-branch <branch>     Branch name for event weight [default: Weight]
    --silent                     Silent mode (disable output)
    -h --help                    Show this help message

Notes:
    - The program will exit if either <data> or <accmc> file does not exist or does
      not end with '.root'.
"""

from docopt import docopt
from rich.console import Console
from rich.table import Table
import pykmcmc.reader
import pykmcmc.likelihood
from pathlib import Path
import sys
import scipy.optimize as opt
from codetiming import Timer
import numpy as np
import matplotlib.pyplot as plt


def optimize():
    args = docopt(__doc__)
    console = Console(quiet=args['--silent'], record=True)
    error_console = Console(stderr=True, style="bold red")
    data_path = Path(args['<data>'])
    if not data_path.exists():
        error_console.print(f"File not found: {data_path}")
        sys.exit(1)
    if not data_path.suffix == '.root':
        error_console.print(f"Data path must be a .root file ({data_path})")
        sys.exit(1)
    accmc_path = Path(args['<accmc>'])
    if not accmc_path.exists():
        error_console.print(f"File not found: {accmc_path}")
        sys.exit(1)
    if not accmc_path.suffix == '.root':
        error_console.print(f"Monte Carlo path must be a .root file ({accmc_path})")
        sys.exit(1)

    console.rule("Loading ROOT files")
    branch_table = Table(title='Branch Info')
    branch_table.add_column("Variable Name")
    branch_table.add_column("Branch Name")
    branch_table.add_row("Mass", args['--mass-branch'])
    branch_table.add_row("Cos(θ)", args['--cos-theta-branch'])
    branch_table.add_row("ϕ", args['--phi-branch'])
    branch_table.add_row("Weight", args['--weight-branch'])
    console.print(branch_table)
    # Set number of parameters
    ndim = 23
    nwalkers = int(args['--walkers'])
    steps = int(args['--steps'])
    
    # Load in the data file and accepted Monte Carlo
    with console.status("Loading DATA..."):
        data = pykmcmc.reader.read_file(data_path,
                                        args['--mass-branch'],
                                        args['--cos-theta-branch'],
                                        args['--phi-branch'],
                                        args['--weight-branch'])
        console.print("Loaded DATA")
    with console.status("Loading ACCMC..."):
        accmc = pykmcmc.reader.read_file(accmc_path,
                                         args['--mass-branch'],
                                         args['--cos-theta-branch'],
                                         args['--phi-branch'],
                                         args['--weight-branch'])
        console.print("Loaded ACCMC")

    ngen = args['--ngen']
    if not ngen:
        ngen = accmc.N

    # Create our likelihood function
    likelihood = pykmcmc.likelihood.Likelihood(data, accmc, ngen)
    func = lambda x: likelihood.negative_log_likelihood(x)
    ndim = 23
    bounds = list(zip([-500]*ndim, [500]*ndim))
    results = dict()


    def callback(xk, *args, **kwargs):
        pass
        # console.print(f"New best minimum: {xk}")
        # console.print(args)
        # console.print(kwargs)
        # console.rule()

    console.rule("Differential Evolution")
    with Timer(name="timer", text="Time to run Differential Evolution: {milliseconds:.0f} ms"):
        results['DE'] = opt.differential_evolution(func, bounds, callback=callback)
    console.print(results['DE'])
    console.print(results['DE'].x)
    plot(results, 'DE', data, accmc)

    console.rule("DIRECT")
    with Timer(name="timer", text="Time to run DIRECT: {milliseconds:.0f} ms"):
        results['DIRECT'] = opt.direct(func, bounds, callback=callback, maxfun=10_000 * ndim, f_min=0.0)
    console.print(results['DIRECT'])
    console.print(results['DIRECT'].x)
    plot(results, 'DIRECT', data, accmc)

    console.rule("Dual Annealing")
    with Timer(name="timer", text="Time to run Dual Annealing: {milliseconds:.0f} ms"):
        results['DA'] = opt.dual_annealing(func, bounds, callback=callback)
    plot(results, 'DA', data, accmc)
    console.print(results['DA'])
    console.print(results['DA'].x)

    console.rule("SHGO")
    with Timer(name="timer", text="Time to run SHG Optimization: {milliseconds:.0f} ms"):
        results['SHGO'] = opt.shgo(func, bounds, callback=callback)
    plot(results, 'SHGO', data, accmc)
    console.print(results['SHGO'])
    console.print(results['SHGO'].x)

def plot(results, key, data, accmc):
    if results[key].success:
        best_pars = results[key].x
        fit_weights = accmc.calc_weights(best_pars)
        fw_sum = np.sum(fit_weights)
        dw_sum = np.sum(data.weight)
        bins = 70
        f0_weights = accmc.calc_weights_f0(best_pars)
        f2_weights = accmc.calc_weights_f2(best_pars)
        a0_weights = accmc.calc_weights_a0(best_pars)
        a2_weights = accmc.calc_weights_a2(best_pars)
        plt.hist(data.mass, weights=data.weight, bins=bins, histtype='step', label='Data')
        plt.hist(accmc.mass, weights=fit_weights / fw_sum * dw_sum, bins=bins, histtype='step', label=f'Fit {results[key].fun}')
        plt.hist(accmc.mass, weights=f0_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit f0')
        plt.hist(accmc.mass, weights=f2_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit f2')
        plt.hist(accmc.mass, weights=a0_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit a0')
        plt.hist(accmc.mass, weights=a2_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit a2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{key}_optimization.svg")
