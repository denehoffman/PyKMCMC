from typing import List, Union

import numpy as np
from emcee.model import Model
from emcee.moves.move import Move
from emcee.state import State

__all__ = ["DREAMMove", "SliceMove"]


class DREAMMove(Move):
    def __init__(
        self,
        b=0.1,
        b_star=0.01,
        jump=False,
        n_cr=1,
        burn_in=0,
        delta=1,
        live_dangerously=False,
        randomize_split=True,
    ):
        self.b = b
        self.b_star = b_star
        self.delta = delta
        if jump:
            self.gamma = lambda _: 1.0
        else:
            self.gamma = lambda d_prime: 2.38 / np.sqrt(2 * self.delta * d_prime)
        self.n_cr = n_cr
        self.L = np.zeros(self.n_cr)
        self.Delta = np.zeros(self.n_cr)
        self.p = np.ones(self.n_cr) / self.n_cr
        self.nsplits = 3
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split
        self.burn_in = burn_in
        self.t = 1

    def setup(self, coords):
        pass

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        for split in range(self.nsplits):
            S1 = inds == split

            # Get the two halves of the ensemble.
            sets = [state.coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = sets[:split] + sets[split + 1 :]

            # Get the move-specific proposal.
            q, ms = self.get_proposal(s, c, model.random)

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            # Loop over the walkers and update them accordingly.
            for i, (j, nlp) in enumerate(zip(all_inds[S1], new_log_probs)):
                lnpdiff = nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            if self.t <= self.burn_in:
                var = np.var(new_state.coords, axis=1)
                delta_Delta = np.sum(np.square(new_state.coords - s) / var, axis=1)
                for i, m in enumerate(ms):
                    self.Delta[m] += delta_Delta[i]
                self.p = (
                    self.t * self.burn_in * (self.Delta / self.L) / np.sum(self.Delta)
                )
                self.t += 1
            state = self.update(state, new_state, accepted, S1)

        return state, accepted

    def get_proposal(self, s, c, random):
        Ns = len(s)  # length of this ensemble split
        d = s.shape[1]  # dimension of parameter space
        q = np.empty((Ns, d), dtype=np.float64)  # create new empty proposal
        ms = np.empty(Ns, dtype=np.int64)
        for i in range(Ns):  # for each chain
            # for delta = 1 this is just DE:
            r1 = np.sum(random.choice(c[0], self.delta, replace=False), axis=0)
            r2 = np.sum(random.choice(c[1], self.delta, replace=False), axis=0)
            e_vec = random.uniform(-self.b, self.b, size=d)
            ep_vec = random.normal(0, self.b_star, size=d)
            m = np.argmax(random.multinomial(1, self.p))
            CR = (m + 1) / self.n_cr
            self.L[m] += 1
            rand_U = random.uniform(0, 1, size=d)
            crossed_indices = rand_U <= CR  # I think eq 5 is wrong
            d_prime = np.sum(crossed_indices)
            s_prime = (
                s[i] + (np.ones(d) + e_vec) * self.gamma(d_prime) * (r2 - r1) + ep_vec
            )
            proposal = np.where(crossed_indices, s_prime, s[i])
            ms[i] = m
            q[i] = proposal
        return q, ms


class SliceMove(Move):
    def __init__(self, w: Union[float, List[float]], max_evals: int = 100):
        self.w = w
        self.max_evals = max_evals

    def propose(self, model: Model, state: State):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if isinstance(self.w, float):
            self.w = [self.w] * ndim
        width = self.w[:]
        accepted = np.zeros(nwalkers, dtype=bool)
        log_probs, new_blobs = model.compute_log_prob_fn(state.coords)
        q = np.empty((nwalkers, ndim), dtype=np.float64)  # create new empty proposal
        for walker in range(nwalkers):
            # get coordinate
            x = state.coords[walker]
            # get g(x) = log(f(x))
            g_x = log_probs[walker]
            while not accepted[walker]:
                # draw z = log(y) where y ~ U(0, f(x))
                z_aux = g_x - model.random.exponential()
                # create hyperrectangle
                L = np.zeros(ndim, dtype=float)
                R = np.zeros(ndim, dtype=float)
                for i in range(ndim):
                    u = model.random.uniform(0, 1)
                    L[i] = x[i] - width[i] * u
                    R[i] = L[i] + width[i]
                for n_eval in range(self.max_evals):
                    q[walker] = model.random.uniform(L, R)
                    g_x = model.log_prob_fn(q[walker])
                    if z_aux <= g_x:
                        accepted[walker] = True
                        break
                    print(f"Resizing slice, attempt {n_eval + 1}")
                    for i in range(ndim):
                        if q[walker][i] < x[i]:
                            L[i] = q[walker][i]
                        else:
                            R[i] = q[walker][i]
                if not accepted[walker]:
                    width *= 2
                    print(f"Step failed! Retrying with w = {width}")
        # this next part is horribly inefficient because I have to calculate the log-probs again
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)
        return state, accepted
