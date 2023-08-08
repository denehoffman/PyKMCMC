from jax import jit, tree_util
import jax.numpy as jnp

class Amplitude:
    def __init__(self,
                 g_rc,
                 m_r,
                 c_cc,
                 m1_c,
                 m2_c):
        self.g_rc = jnp.array(g_rc)
        self.m_r = jnp.array(m_r)
        self.c_cc = jnp.array(c_cc)
        self.m1_c = jnp.array(m1_c)
        self.m2_c = jnp.array(m2_c)
        self.n_resonances, self.n_channels = self.g_rc.shape
        self.g_rcc = jnp.expand_dims(self.g_rc, axis=-1)
        msq_r = jnp.square(self.m_r)
        self.msq_rc = jnp.expand_dims(msq_r, axis=-1)
        self.msq_rcc = jnp.expand_dims(self.msq_rc, axis=-1)
        self.c_rcc = jnp.expand_dims(self.c_cc, axis=-1).T
        self.m1_rc = jnp.expand_dims(self.m1_c, axis=-1).T
        self.m2_rc = jnp.expand_dims(self.m2_c, axis=-1).T
        self.res_chi_plus_rc = 1 - jnp.square(self.m1_rc + self.m2_rc) / self.msq_rc
        self.res_chi_minus_rc = 1 - jnp.square(self.m1_rc - self.m2_rc) / self.msq_rc
        self.res_rho_rc = jnp.sqrt(self.res_chi_plus_rc * self.res_chi_plus_rc)
        self.eye_cc = jnp.eye(self.n_channels)

    @staticmethod
    def pars_to_cx_beta_r(*pars):
        pars = list(pars)
        pars.insert(0, 0.0)  # f_0(500) re
        pars.insert(0, 0.0)  # f_0(500) im
        pars.insert(3, 0.0)  # f_0(980) im
        betas = jnp.array([par_re + 1j * par_im for par_re, par_im in zip(pars[::2], pars[1::2])])
        return betas
    
    @staticmethod
    @jit
    def calculate(IKC_inv_vector, P_vector):
        return IKC_inv_vector @ P_vector

    def _tree_flatten(self):
        children = (self.g_rc, self.m_r, self.c_cc, self.m1_c, self.m2_c)
        aux_data = dict()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

tree_util.register_pytree_node(Amplitude, Amplitude._tree_flatten, Amplitude._tree_unflatten)


class AmplitudeS(Amplitude):
    @jit
    def K_cc(self, s):
        num = self.g_rcc * jnp.transpose(self.g_rcc, (0, 2, 1))
        den = self.msq_rcc - s
        internal = num / den + self.c_rcc
        return jnp.sum(internal, axis=0)

    @jit
    def chi_plus_c(self, s):
        num = jnp.square(self.m1_c + self.m2_c)
        return 1 - num / s

    @jit
    def chi_minus_c(self, s):
        num = jnp.square(self.m1_c - self.m2_c)
        return 1 - num / s

    @jit
    def rho_c(self, s):
        return jnp.sqrt(self.chi_plus_c(s) * self.chi_minus_c(s) + 0j)

    @jit
    def C_cc(self, s):
        term_1_c = (self.rho_c(s) / jnp.pi
                    * jnp.log((self.chi_plus_c(s) + self.rho_c(s))
                             / (self.chi_plus_c(s) - self.rho_c(s))))
        term_2_c = (self.chi_plus_c(s) / jnp.pi
                    * (self.m2_c - self.m1_c)
                    / (self.m1_c + self.m2_c)
                    * jnp.log(self.m2_c / self.m1_c))
        diag_c = term_1_c + term_2_c
        return jnp.diag(diag_c)
    
    @jit
    def IKC_cc(self, s):
        KC_cc = self.K_cc(s) @ self.C_cc(s)
        return self.eye_cc + KC_cc

    @jit
    def IKC_inv_c(self, s, channel=0):
        IKC_inv_cc = jnp.linalg.inv(self.IKC_cc(s))
        return IKC_inv_cc[channel]

    @jit
    def P_c(self, s, cx_beta_r):
        cx_beta_rc = jnp.expand_dims(cx_beta_r, axis=-1)
        num = cx_beta_rc * self.g_rc
        denom = self.msq_rc - s
        internal = num / denom
        return jnp.sum(internal, axis=0)


tree_util.register_pytree_node(AmplitudeS, AmplitudeS._tree_flatten, AmplitudeS._tree_unflatten)


class AmplitudeAdlerS(AmplitudeS):
    def __init__(self,
                 g_rc,
                 m_r,
                 c_cc,
                 m1_c,
                 m2_c,
                 s_0,
                 s_norm):
        super().__init__(g_rc, m_r, c_cc, m1_c, m2_c)
        self.s_0 = s_0
        self.s_norm = s_norm

    @jit
    def K_cc(self, s):
        num = self.g_rcc * jnp.transpose(self.g_rcc, (0, 2, 1))
        den = self.msq_rcc - s
        internal = num / den + self.c_rcc
        return ((s - self.s_0) / self.s_norm) * jnp.sum(internal, axis=0)

    def _tree_flatten(self):
        children = (self.g_rc, self.m_r, self.c_cc, self.m1_c, self.m2_c, self.s_0, self.s_norm)
        aux_data = dict()
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

tree_util.register_pytree_node(AmplitudeAdlerS, AmplitudeAdlerS._tree_flatten, AmplitudeAdlerS._tree_unflatten)

class AmplitudeD(AmplitudeS):
    def __init__(self,
                 g_rc,
                 m_r,
                 c_cc,
                 m1_c,
                 m2_c):
        super().__init__(g_rc, m_r, c_cc, m1_c, m2_c)
        res_q_rc = self.res_rho_rc * jnp.sqrt(self.msq_rc) / 2
        self.res_z_rc = jnp.square(res_q_rc) / (0.1973 * 0.1973)
        self.res_bw_rc = AmplitudeD.bw(self.res_z_rc)

    @jit
    def z_c(self, s):
        q = self.rho_c(s) * jnp.sqrt(s) / 2
        return jnp.square(q) / (0.1973 * 0.1973)

    @staticmethod
    @jit
    def bw(z):
        return jnp.sqrt(13 * jnp.square(z) / (jnp.square(z - 3) + 9 * z))

    @jit
    def blatt_weisskopf_rc(self, s):
        num_c = AmplitudeD.bw(self.z_c(s))
        num_rc = jnp.expand_dims(num_c, axis=-1).T
        return num_rc / self.res_bw_rc

    @jit
    def K_cc(self, s):
        num = self.g_rcc * jnp.transpose(self.g_rcc, (0, 2, 1))
        den = self.msq_rcc - s
        internal = num / den + self.c_rcc
        bw_rc = self.blatt_weisskopf_rc(s)
        bw_right_rcc = jnp.expand_dims(bw_rc, axis=-1)
        bw_left_rcc = jnp.transpose(bw_right_rcc, (0, 2, 1))
        return jnp.sum(bw_left_rcc * internal * bw_right_rcc, axis=0)

    @jit
    def P_c(self, s, cx_beta_r, bw_rc):
        cx_beta_rc = jnp.expand_dims(cx_beta_r, axis=-1)
        num = cx_beta_rc * self.g_rc
        denom = self.msq_rc - s
        internal = num / denom
        return jnp.sum(internal * bw_rc, axis=0)

tree_util.register_pytree_node(AmplitudeD, AmplitudeD._tree_flatten, AmplitudeD._tree_unflatten)

f0_m1_c = [0.13498, 0.26995, 0.49368, 0.54786, 0.54786]
f0_m2_c = [0.13498, 0.26995, 0.49761, 0.54786, 0.95778]
f0_m_r = [0.51461, 0.90630, 1.23089, 1.46104, 1.69611]
f0_g_rc = [
        [+0.74987, -0.01257, +0.02736, -0.15102, +0.36103],
        [+0.06401, +0.00204, +0.77413, +0.50999, +0.13112],
        [-0.23417, -0.01032, +0.72283, +0.11934, +0.36792],
        [+0.01570, +0.26700, +0.09214, +0.02742, -0.04025],
        [-0.14242, +0.22780, +0.15981, +0.16272, -0.17397]
        ]
f0_c_cc = [
        [+0.03728, +0.00000, -0.01398, -0.02203, +0.01397],
        [+0.00000, +0.00000, +0.00000, +0.00000, +0.00000],
        [-0.01398, +0.00000, +0.02349, +0.03101, -0.04003],
        [-0.02203, +0.00000, +0.03101, -0.13769, -0.06722],
        [+0.01397, +0.00000, -0.04003, -0.06722, -0.28401]
        ]

f2_m1_c = [0.13498, 0.26995, 0.49368, 0.54786]
f2_m2_c = [0.13498, 0.26995, 0.49761, 0.54786]
f2_m_r = [1.15299, 1.48359, 1.72923, 1.96700]
f2_g_rc = [
        [+0.40033, +0.15479, -0.08900, -0.00113],
        [+0.01820, +0.17300, +0.32393, +0.15256],
        [-0.06709, +0.22941, -0.43133, +0.23721],
        [-0.49924, +0.19295, +0.27975, -0.03987]
        ]
f2_c_cc = [
        [-0.04319, +0.00000, +0.00984, +0.01028],
        [+0.00000, +0.00000, +0.00000, +0.00000],
        [+0.00984, +0.00000, -0.07344, +0.05533],
        [+0.01028, +0.00000, +0.05533, -0.05183]
        ]

a0_m1_c = [0.13498, 0.49368]
a0_m2_c = [0.54786, 0.49761]
a0_m_r = [0.95395, 1.26767]
a0_g_rc = [
        [+0.43215, -0.28825],
        [+0.19000, +0.43372]
        ]
a0_c_cc = [
        [+0.00000, +0.00000],
        [+0.00000, +0.00000],
        ]

a2_m1_c = [0.49368, 0.13498, 0.13498]
a2_m2_c = [0.54786, 0.49761, 0.95778]
a2_m_r = [1.30080, 1.75351]
a2_g_rc = [
        [+0.30073, +0.21426, -0.09162],
        [+0.68567, +0.12543, +0.00184]
        ]
a2_c_cc = [
        [-0.40184, +0.00033, -0.08707],
        [+0.00033, -0.21416, -0.06193],
        [-0.08707, -0.06193, -0.17435]
        ]

f0 = AmplitudeAdlerS(f0_g_rc,
                     f0_m_r,
                     f0_c_cc,
                     f0_m1_c,
                     f0_m2_c,
                     s_0=0.0091125,
                     s_norm=1.0)
f2 = AmplitudeD(f2_g_rc,
                f2_m_r,
                f2_c_cc,
                f2_m1_c,
                f2_m2_c)
a0 = AmplitudeS(a0_g_rc,
                a0_m_r,
                a0_c_cc,
                a0_m1_c,
                a0_m2_c)
a2 = AmplitudeD(a2_g_rc,
                a2_m_r,
                a2_c_cc,
                a2_m1_c,
                a2_m2_c)
