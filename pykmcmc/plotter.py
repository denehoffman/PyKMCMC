"""
Usage:
    kplot [options] trace <chain.h5>
    kplot [options] corner <chain.h5>
    kplot [options] best <chain.h5> <data> <accmc>

Options:
    -f --follow  Plot live data
    --burn <n>  Number of burned-in steps
    --mass-branch <branch>       Branch name for mass [default: M_FinalState]
    --cos-theta-branch <branch>  Branch name for cos theta [default: HX_CosTheta]
    --phi-branch <branch>        Branch name for phi [default: HX_Phi]
    --weight-branch <branch>     Branch name for event weight [default: Weight]
"""

from docopt import docopt
import numpy as np
import arviz as az
import arviz.labels as azl
import prism
from pathlib import Path
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pykmcmc.reader
from rich.console import Console

latex_labels = [
        r"$f_0(980)\ \Re$",
        r"$f_0(1370)\ \Re$",
        r"$f_0(1370)\ \Im$",
        r"$f_0(1500)\ \Re$",
        r"$f_0(1500)\ \Im$",
        r"$f_0(1710)\ \Re$",
        r"$f_0(1710)\ \Im$",
        r"$f_2(1270)\ \Re$",
        r"$f_2(1270)\ \Im$",
        r"$f_2'(1525)\ \Re$",
        r"$f_2'(1525)\ \Im$",
        r"$f_2(1810)\ \Re$",
        r"$f_2(1810)\ \Im$",
        r"$f_2(1950)\ \Re$",
        r"$f_2(1950)\ \Im$",
        r"$a_0(980)\ \Re$",
        r"$a_0(980)\ \Im$",
        r"$a_0(1450)\ \Re$",
        r"$a_0(1450)\ \Im$",
        r"$a_2(1320)\ \Re$",
        r"$a_2(1320)\ \Im$",
        r"$a_2(1700)\ \Re$",
        r"$a_2(1700)\ \Im$",
        ]
var_names = [
        "f0980Re",
        "f01370Re",
        "f01370Im",
        "f01500Re",
        "f01500Im",
        "f01710Re",
        "f01710Im",
        "f21270Re",
        "f21270Im",
        "f21525Re",
        "f21525Im",
        "f21810Re",
        "f21810Im",
        "f21950Re",
        "f21950Im",
        "a0980Re",
        "a0980Im",
        "a01450Re",
        "a01450Im",
        "a21320Re",
        "a21320Im",
        "a21700Re",
        "a21700Im",
        ]
console = Console()
error_console = Console(stderr=True, style="bold red")

def to_arviz(filename):
    reader = emcee.backends.HDFBackend(filename)
    df = az.from_emcee(reader, var_names=var_names)
    return df, reader

def plotter():
    args = docopt(__doc__)
    filename = args['<chain.h5>']
    filepath = Path(filename)
    if not filepath.exists():
        print("File not found!")
        return
    if args['corner']:
        plot_corner(filepath, args)
    if args['trace']:
        plot_trace(filepath, args)
    if args['best']:
        plot_best(filepath, args)

def plot_best(filepath, args):
    data_path = Path(args['<data>'])
    if not data_path.exists():
        error_console.print(f"File not found: {data_path}")
        return
    if not data_path.suffix == '.root':
        error_console.print(f"Data path must be a .root file ({data_path})")
        return
    accmc_path = Path(args['<accmc>'])
    if not accmc_path.exists():
        error_console.print(f"File not found: {accmc_path}")
        return
    if not accmc_path.suffix == '.root':
        error_console.print(f"Monte Carlo path must be a .root file ({accmc_path})")
        return
    data = pykmcmc.reader.read_file(data_path,
                                    args['--mass-branch'],
                                    args['--cos-theta-branch'],
                                    args['--phi-branch'],
                                    args['--weight-branch'])
    accmc = pykmcmc.reader.read_file(accmc_path,
                                         args['--mass-branch'],
                                         args['--cos-theta-branch'],
                                         args['--phi-branch'],
                                         args['--weight-branch'])
    df, sampler = to_arviz(filepath)
    chain = sampler.get_chain(flat=True)
    lls = sampler.get_log_prob(flat=True)
    chain = sampler.get_chain(flat=True)
    i_best = np.argmax(lls)
    best_pars = chain[i_best]
    fit_weights = accmc.calc_weights(best_pars)
    fw_sum = np.sum(fit_weights)
    dw_sum = np.sum(data.weight)
    bins = 70
    f0_weights = accmc.calc_weights_f0(best_pars)
    f2_weights = accmc.calc_weights_f2(best_pars)
    a0_weights = accmc.calc_weights_a0(best_pars)
    a2_weights = accmc.calc_weights_a2(best_pars)
    plt.hist(data.mass, weights=data.weight, bins=bins, histtype='step', label='Data')
    plt.hist(accmc.mass, weights=fit_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit')
    plt.hist(accmc.mass, weights=f0_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit f0')
    plt.hist(accmc.mass, weights=f2_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit f2')
    plt.hist(accmc.mass, weights=a0_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit a0')
    plt.hist(accmc.mass, weights=a2_weights / fw_sum * dw_sum, bins=bins, histtype='step', label='Fit a2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_corner(filepath, args):
    def make_plot(filepath, burn):
        with az.rc_context(rc={'plot.max_subplots': None}):
            df, sampler = to_arviz(filepath)
            df_burned = df.sel(draw=slice(burn, None))
            lls = sampler.get_log_prob(flat=True)
            chain = sampler.get_chain(flat=True)
            i_best = np.argmax(lls)
            best_pars = chain[i_best]
            labeller = azl.MapLabeller(var_name_map={k: v for k, v in zip(var_names, latex_labels)})
            az.plot_pair(df_burned, labeller=labeller, kind='kde')
            plt.tight_layout()
    if args['--follow']:
        ani = FuncAnimation(plt.gcf(), lambda _: make_plot(filepath, args['--burn']), interval=1000)
        plt.show()
    else:
        make_plot(filepath, args['--burn'])
        plt.show()

# def plot_trace(filepath, burn, live):
#     fig = plt.figure()
#     def make_plot(filepath, burn, selected_names):
#         with az.rc_context(rc={'plot.max_subplots': None}):
#             df, sampler = to_arviz(filepath)
#             df_burned = df.sel(draw=slice(burn, None))
#             lls = sampler.get_log_prob(flat=True)
#             chain = sampler.get_chain(flat=True)
#             i_best = np.argmax(lls)
#             best_pars = chain[i_best]
#             labeller = azl.MapLabeller(var_name_map={k: v for k, v in zip(var_names, latex_labels)})
#             # Draw figure
#             axes = az.plot_trace(df_burned, labeller=labeller, combined=True, var_names=selected_names)
#             if len(selected_names) > 1 and not isinstance(selected_names, str):
#                 for i in range(len(selected_names)):
#                     axes[i, 0].axvline(best_pars[i])
#             else:
#                 axes[0, 0].axvline(best_pars[0])
#             fig.gca().relim()
#             fig.gca().autoscale_view()

#     if live:
#         ani = FuncAnimation(fig, lambda _: make_plot(filepath, burn, var_names[0]), interval=1000, cache_frame_data=False)
#         plt.show()
#     else:
#         make_plot(filepath, burn, var_names[0])
#         plt.show()

def plot_trace(filepath, args):
    df, sampler = to_arviz(filepath)
    df_burned = df.sel(draw=slice(args['--burn'], None))
    labeller = azl.MapLabeller(var_name_map={k: v for k, v in zip(var_names, latex_labels)})
    selected_vars = var_names[:3]
    fig, axes = plt.subplots(len(selected_vars), 1, figsize=(6, 2 * len(selected_vars)), sharex=True)
    if args['--burn']:
        def update(frame):
            df, sampler = to_arviz(filepath)
            df_burned = df.sel(draw=slice(args['--burn'], None))
            az.plot_trace(df_burned, labeller=labeller, combined=True, var_names=selected_vars)
        ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
        plt.show()
    else:
        az.plot_trace(df_burned, labeller=labeller, combined=True, var_names=selected_vars)
        plt.show()
