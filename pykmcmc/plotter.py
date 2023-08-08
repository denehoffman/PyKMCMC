import numpy as np
import arviz as az
import arviz.labels as azl
import prism
import emcee
import corner
import matplotlib.pyplot as plt

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

def to_arviz(filename):
    reader = emcee.backends.HDFBackend(filename)
    df = az.from_emcee(reader, var_names=var_names)
    return df, reader

def plot_mcmc():
    with az.rc_context(rc={'plot.max_subplots': None}):
        df, sampler = to_arviz('demo.h5')
        df_burned = df.sel(draw=slice(200, None))
        lls = sampler.get_log_prob(flat=True)
        chain = sampler.get_chain(flat=True)
        i_best = np.argmax(lls)
        best_pars = chain[i_best]
        labeller = azl.MapLabeller(var_name_map={k: v for k, v in zip(var_names, latex_labels)})
        # corner.corner(df, var_names=var_names[:3])
        axes = az.plot_trace(df, labeller=labeller, combined=True)
        for i in range(23):
            axes[i, 0].axvline(best_pars[i])
        plt.tight_layout()
        plt.savefig('test.svg')

        axes = az.plot_trace(df_burned, labeller=labeller, combined=True)
        for i in range(23):
            axes[i, 0].axvline(best_pars[i])
        plt.tight_layout()
        plt.savefig('test_burned.svg')

        az.plot_pair(df_burned, labeller=labeller, kind='kde')
        plt.tight_layout()
        plt.savefig('test_corner.svg')
