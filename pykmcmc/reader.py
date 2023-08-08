import uproot
import jax.numpy as jnp
from jax import jit, tree_util, vmap
import pykmcmc.amplitude as amp
from codetiming import Timer

def read_file(path,
              mass_branch,
              costheta_branch,
              phi_branch,
              weight_branch):
    with uproot.open(path) as tfile:
        ttree = tfile.get(tfile.keys()[0])
        with Timer(name="timer", text="Time to read tree: {milliseconds:.0f} ms"):
            df = ttree.arrays([mass_branch,
                               costheta_branch,
                               phi_branch,
                               weight_branch],
                              library='np')
        return Dataset(jnp.array(df[mass_branch]),
                       jnp.array(df[costheta_branch]),
                       jnp.array(df[phi_branch]),
                       jnp.array(df[weight_branch]))

class Dataset:
    def __init__(self, mass, costheta, phi, weight):
        self.N = len(mass)
        self.mass = mass
        self.costheta = costheta
        self.phi = phi
        self.weight = weight
        with Timer(name="timer", text="Time to precompute: {milliseconds:.0f} ms"):
            self.f0_ikc_inv = vmap(lambda m: amp.f0.IKC_inv_c(m**2, channel=2))(mass)
            self.f2_ikc_inv = vmap(lambda m: amp.f2.IKC_inv_c(m**2, channel=2))(mass)
            self.a0_ikc_inv = vmap(lambda m: amp.a0.IKC_inv_c(m**2, channel=1))(mass)
            self.a2_ikc_inv = vmap(lambda m: amp.a2.IKC_inv_c(m**2, channel=1))(mass)
            self.f2_bw_rc = vmap(lambda m: amp.f2.blatt_weisskopf_rc(m**2))(mass)
            self.a2_bw_rc = vmap(lambda m: amp.a2.blatt_weisskopf_rc(m**2))(mass)
            self.d_wave = vmap(self.single_d_wave)(self.phi, self.costheta)

    @staticmethod
    @jit
    def single_d_wave(phi, costheta):
        return jnp.sqrt(15 / (2 * jnp.pi)) * jnp.exp(2j * phi) * (1 - jnp.square(costheta)) / 4

    @staticmethod
    @jit
    def single_calc(cx_beta_r,
                    mass,
                    f0_ikc_inv,
                    f2_ikc_inv,
                    a0_ikc_inv,
                    a2_ikc_inv,
                    f2_bw_rc,
                    a2_bw_rc,
                    d_wave,
                    weight):
        f0_P_c = amp.f0.P_c(mass**2, cx_beta_r[0:5])
        f2_P_c = amp.f2.P_c(mass**2, cx_beta_r[5:9], f2_bw_rc)
        a0_P_c = amp.a0.P_c(mass**2, cx_beta_r[9:11])
        a2_P_c = amp.a2.P_c(mass**2, cx_beta_r[11:13], a2_bw_rc)
        f0_term = amp.Amplitude.calculate(f0_ikc_inv, f0_P_c)
        f2_term = amp.Amplitude.calculate(f2_ikc_inv, f2_P_c)
        a0_term = amp.Amplitude.calculate(a0_ikc_inv, a0_P_c)
        a2_term = amp.Amplitude.calculate(a2_ikc_inv, a2_P_c)
        s_wave = jnp.sqrt(1 / (4 * jnp.pi))
        return weight * jnp.nan_to_num(jnp.abs(s_wave * (f0_term + a0_term)
                                               + d_wave * (f2_term + a2_term))**2)

    def calc(self, cx_beta_r):
        return jnp.sum(vmap(Dataset.single_calc,
                            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(cx_beta_r,
                                                                       self.mass,
                                                                       self.f0_ikc_inv,
                                                                       self.f2_ikc_inv,
                                                                       self.a0_ikc_inv,
                                                                       self.a2_ikc_inv,
                                                                       self.f2_bw_rc,
                                                                       self.a2_bw_rc,
                                                                       self.d_wave,
                                                                       self.weight))

    @staticmethod
    def single_calc_log(cx_beta_r,
                        mass,
                        f0_ikc_inv,
                        f2_ikc_inv,
                        a0_ikc_inv,
                        a2_ikc_inv,
                        f2_bw_rc,
                        a2_bw_rc,
                        d_wave,
                        weight):
        f0_P_c = amp.f0.P_c(mass**2, cx_beta_r[0:5])
        f2_P_c = amp.f2.P_c(mass**2, cx_beta_r[5:9], f2_bw_rc)
        a0_P_c = amp.a0.P_c(mass**2, cx_beta_r[9:11])
        a2_P_c = amp.a2.P_c(mass**2, cx_beta_r[11:13], a2_bw_rc)
        f0_term = amp.Amplitude.calculate(f0_ikc_inv, f0_P_c)
        f2_term = amp.Amplitude.calculate(f2_ikc_inv, f2_P_c)
        a0_term = amp.Amplitude.calculate(a0_ikc_inv, a0_P_c)
        a2_term = amp.Amplitude.calculate(a2_ikc_inv, a2_P_c)
        s_wave = jnp.sqrt(1 / (4 * jnp.pi))
        return weight * jnp.nan_to_num(jnp.log(jnp.abs(s_wave * (f0_term + a0_term)
                                                       + d_wave * (f2_term + a2_term))**2))

    def calc_log(self, cx_beta_r):
        log_values = vmap(Dataset.single_calc_log,
                          in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(cx_beta_r,
                                                                     self.mass,
                                                                     self.f0_ikc_inv,
                                                                     self.f2_ikc_inv,
                                                                     self.a0_ikc_inv,
                                                                     self.a2_ikc_inv,
                                                                     self.f2_bw_rc,
                                                                     self.a2_bw_rc,
                                                                     self.d_wave,
                                                                     self.weight)
        return jnp.sum(log_values)

    def _tree_flatten(self):
        children = (self.mass, self.costheta, self.phi, self.weight)
        aux_data = dict()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

tree_util.register_pytree_node(Dataset, Dataset._tree_flatten, Dataset._tree_unflatten)
