"""
Usage:
    kmcmc [options] <data> <accmc>

Options:
    -o --output <output>         Output file name ending with '.h5' [default: chain.h5]
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
    - The move config should be specified with the following (example) structure:
        [[move]]
        name = "DEMove"  # name of move in emcee.moves
        parameters = { sigma = 1e-3 }  # parameter overrides (leave = {} for default)
        weight = 0.4  # probability for emcee to use this move
        
        [[move]]
        ...
      and so on.

Examples:
    kmcmc data.root accmc.root -o mychain.h5 -s 100
    - Runs 100 steps

    mpiexec -n 4 kmcmc data.root accmc.root --mpi
    - Run using OpenMPI on 4 cores

    kmcmc data.root accmc.root --ncores 4
    - Run using Python multiprocessing on 4 cores
"""


import os
import sys
import emcee
from jax import random
import pykmcmc.likelihood
import pykmcmc.reader
from rich.console import Console
from rich.table import Table
import numpy as np
from pathlib import Path
from datetime import datetime
import schwimmbad
from docopt import docopt
try:
    import tomllib as toml
except ImportError:
    import tomli as toml

def load_moves(config_path):
    moves = []
    config = toml.loads(config_path.read_text())['move']
    for move_data in config:
        move_name = move_data['name']
        move_class = getattr(emcee.moves, move_name)
        move_params = move_data['parameters']
        move_weight = move_data['weight']
        move_instance = move_class(**move_params)
        moves.append((move_instance, move_weight))
    return moves

def run_mcmc():
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
    output_path = Path(args['--output'])
    if output_path.exists() and not args['--force']:
        error_console.print(f"The file {output_path} already exists, use --force to override")
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

    # Setup the backend for saving the chain
    backend = emcee.backends.HDFBackend(output_path)
    backend.reset(nwalkers, ndim)  # clear the file if it exists

    # Let's start the sample as an L2 ball (n-sphere)
    # p0 = random.normal(key, (nwalkers, ndim)) * 100
    if args['--seed']:
        seed = int(args['--seed'])
    else:
        seed = int(datetime.now().timestamp() * 1_000_000)
    key = random.PRNGKey(seed)
    p0 = random.ball(key, d=ndim, shape=(nwalkers,)) * 100
    
    # Our move list will include a default stretch move
    # for some normal-ish walking, a DE move most of the
    # time for jumping around multimodal distributions,
    # a DE move with large sigma to get out of the lattice,
    # a DE move with gamma0 = 1 which is recommended to
    # cause jumps to unsearched modes, and a DE snooker move
    moves = [
        (emcee.moves.StretchMove(), 0.3),
        (emcee.moves.DEMove(sigma=1e-3), 0.4),
        (emcee.moves.DEMove(sigma=10), 0.1), # de-gridify
        (emcee.moves.DEMove(gamma0=1), 0.1), # jump
        (emcee.moves.DESnookerMove(), 0.1),
    ]
    if args['--config']:
        config_path = Path(args['--config'])
        if config_path.exists():
            console.print(f"Found config: {config_path}")
            try:
                moves = load_moves(config_path)
            except:
                error_console.print("Config file invalid")
        else:
            error_console.print(f"File not found: {config_path}")

    pool = None
    if int(args['--ncores']) > 1 or args['--mpi']:
        os.environ["OMP_NUM_THREADS"] = "1"  # turn off numpy parallelization
        pool = schwimmbad.choose_pool(mpi=args['--mpi'], processes=int(args['--ncores']))
        console.print("Using Multithreading!")

    # A table to display the user's settings and system defaults
    console.rule('MCMC Settings')
    run_table = Table(show_header=False)
    run_table.add_column(justify='left')
    run_table.add_column(justify='right')
    run_table.add_row("Data Path", f"{data_path}")
    run_table.add_row("Monte Carlo Path", f"{accmc_path}")
    run_table.add_row("Generated Monte Carlo Size", f"{ngen}")
    run_table.add_row("Output Path", f"{output_path}")
    run_table.add_row("# Walkers", f"{nwalkers}")
    run_table.add_row("# Steps", f"{steps}")
    if int(args['--ncores']) > 1:
        run_table.add_row("Multithreaded?", f"{bool(pool)} ({int(args['--ncores'])} cores)")
    else:
        run_table.add_row("Multithreaded?", f"{bool(pool)}")
    run_table.add_row("MPI?", f"{args['--mpi']}")
    run_table.add_row("Seed", f"{seed}")
    console.print(run_table)
    
    # A table to display some intermediate info
    mcmc_table = Table(title="MCMC Info")
    mcmc_table.add_column("Step")
    mcmc_table.add_column("Mean 𝜏")
    mcmc_table.add_column("Mean Acceptance Fraction")
    mcmc_table.add_column("Best NLL")
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        likelihood.log_likelihood,
                                        moves=moves,
                                        backend=backend,
                                        pool=pool)
        console.rule("Running MCMC")
        # We also want to monitor the autocorrelation as we go, just for fun
        # I'll check the autocorrelation every ten steps for now
        old_tau = np.inf
        best_nll = np.inf
        for _ in sampler.sample(p0, iterations=steps, progress=True,
                                progress_kwargs=dict(
                                    unit='step',
                                    ascii=['\x1b[38;5;237m━\x1b[0m',
                                           '\x1b[38;5;10m━\x1b[0m'],
                                    dynamic_ncols=True,
                                )):
            # check/log important info every 10 steps
            if sampler.iteration % 10:
                continue
            
            # check the config file and update move list if it has changed!
            if args['--config']:
                config_path = Path(args['--config'])
                if config_path.exists():
                    try:
                        moves = load_moves(config_path)
                    except:
                        error_console.print("Config file invalid")
            sampler._moves, sampler._weights = zip(*moves)


            # grab tau and see if tau * 100 < i_step for all walkers
            tau = sampler.get_autocorr_time(tol=0)
            # check if we have a new best log-likelihood
            lls = sampler.get_log_prob(flat=True)
            best = np.amax(lls)
            if -best < best_nll:
                best_nll = -best
            mcmc_table.add_row(f"{sampler.iteration}",
                               f"{np.mean(tau):.3f}",
                               f"{np.mean(sampler.acceptance_fraction):.0%}",
                               f"{best_nll:.0f}")
            converged = np.all(tau * 100 < sampler.iteration)

            # additionally check to see if tau isn't changing very much
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            # if converged, stop taking new samples
            if converged:
                console.print("A miracle occurred, the MCMC converged!")
                break
            old_tau = tau
    except KeyboardInterrupt:
        error_console.print("Interrupted!")
    finally:
        if pool:
            pool.close()
        console.print(mcmc_table)
        lls = sampler.get_log_prob(flat=True)
        pars = sampler.get_chain(flat=True)
        best_ll = np.amax(lls)
        ibest_ll = np.argmax(lls)
        best_pars = pars[ibest_ll]
        console.print(f"Global Best Negative Log Likelihood: {-best_ll:.0f}")
        labels = ["f_0(980) Re",
                  "f_0(1370) Re",
                  "f_0(1370) Im",
                  "f_0(1500) Re",
                  "f_0(1500) Im",
                  "f_0(1710) Re",
                  "f_0(1710) Im",
                  "f_2(1270) Re",
                  "f_2(1270) Im",
                  "f_2'(1525) Re",
                  "f_2'(1525) Im",
                  "f_2(1810) Re",
                  "f_2(1810) Im",
                  "f_2(1950) Re",
                  "f_2(1950) Im",
                  "a_0(980) Re",
                  "a_0(980) Im",
                  "a_0(1450) Re",
                  "a_0(1450) Im",
                  "a_2(1320) Re",
                  "a_2(1320) Im",
                  "a_2(1700) Re",
                  "a_2(1700) Im"]
        best_pars_dict = {label: par for label, par in zip(labels, best_pars)}
        pars_table = Table(title="Best Minimum")
        pars_table.add_column("Parameter Name", justify='center')
        pars_table.add_column("Parameter Value", justify='center')
        for k, v in best_pars_dict.items():
            pars_table.add_row(f"{k}", f"{v:.3f}")
        console.print(pars_table)
        console.print("Table as dictionary for easy copying:")
        console.print(best_pars_dict)
        console.rule("MCMC Complete")
        console.print(f"Chain written to {output_path}")
        console.save_text("pymcmc_log.txt")
