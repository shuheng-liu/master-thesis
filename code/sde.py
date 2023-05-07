import numpy as np
import matplotlib.pyplot as plt
import torch

from neurodiffeq import diff
from neurodiffeq.conditions import IVP, NoCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.utils import set_seed

import visualization_helper
import sdeint
from scipy.integrate import cumulative_trapezoid

DOMAIN = np.linspace(0, 0.2, 1000)


def polymorphic(torch_f, numpy_f):
    def wrapper(x):
        if isinstance(x, torch.Tensor):
            return torch_f(x)
        else:
            return numpy_f(x)

    return wrapper


sin = polymorphic(torch.sin, np.sin)
cos = polymorphic(torch.cos, np.cos)
exp = polymorphic(torch.exp, np.exp)
log = polymorphic(torch.log, np.log)
sqrt = polymorphic(torch.sqrt, np.sqrt)


def a(t):
    return -(2 + cos(t))


def A(t):
    return -(2 * t + sin(t))


def b(t):
    return exp(-t)


def h(t):
    return log(5 + t)


def h_prime(t):
    return 1 / (5 + t)


def get_sde_solutions(n_realizations):
    return [
        sdeint.itoint(
            f=lambda y, t: a(t) * y + b(t),
            G=lambda y, t: h(t),
            y0=1,
            tspan=DOMAIN,
            generator=np.random.default_rng(seed=i),
        ).flatten()
        for i in range(n_realizations)
    ]


def _l2_norm(f, ts):
    return cumulative_trapezoid(f(ts) ** 2, ts, initial=0) ** 0.5


def get_deterministic_bound(solver, ts):
    residuals = solver.get_residuals(ts, best=False, to_numpy=True)
    return np.exp(A(ts)) * cumulative_trapezoid(abs(residuals) * np.exp(-A(ts)), ts, initial=0)


def get_stochastic_bound(ts, eps1, eps2):
    I1 = abs(h(ts)) * sqrt(2 * ts * np.log(1 / eps1))

    phi = lambda t: exp(-A(t)) * (h_prime(t) - a(t) * h(t))
    I2 = exp(A(ts)) * _l2_norm(phi, ts) * ts * sqrt(0.5 + sqrt(1 / eps2 / 3 - 1 / 3))
    return I1 + I2


if __name__ == "__main__":
    set_seed(0)
    visualization_helper.setup()

    solver = Solver1D(
        lambda u, t: [diff(u, t) - a(t) * u - b(t)],
        [IVP(0, 1)],
        t_min=0, t_max=1,
        n_batches_valid=1,
    )
    solver.fit(1000)
    u = solver.get_solution(best=False)(DOMAIN, to_numpy=True)
    v_realizations = get_sde_solutions(10)
    err_realizations = [u - v for v in v_realizations]

    res = solver.get_residuals(DOMAIN, best=False, to_numpy=True)
    bound_deterministic = get_deterministic_bound(solver, DOMAIN)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), dpi=125)
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        bound_stochastic = get_stochastic_bound(DOMAIN, eps/2, eps/2)
        bound = bound_deterministic + bound_stochastic
        ax.plot(DOMAIN, bound, label=r'$\mathcal{B}_\epsilon(t)$ with $\epsilon = $' + f'{eps:.1f}')
    for err in err_realizations:
        # success_rate = (abs(err) <= bound).mean()
        ax.plot(DOMAIN, abs(err), ':', color='blue', markersize=0.5)
    ax.legend(ncol=2, prop=dict(size=12))
    ax.set_xlabel(r'$t$', fontdict={'size': 16})
    ax.set_ylabel(r'$|u(t) - V_t|$', fontdict={'size': 16})
    plt.show()

    fig.savefig(visualization_helper.get_folder() / 'sde1.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.6), dpi=125)
    eps = 0.2
    for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        bound_stochastic = get_stochastic_bound(DOMAIN, eps * rho, eps * (1-rho))
        bound = bound_deterministic + bound_stochastic
        ax.plot(DOMAIN, bound, label=r'$\mathcal{B}$ with $\rho$=' + f'{rho}')
    for err in err_realizations:
        # success_rate = (abs(err) <= bound).mean()
        ax.plot(DOMAIN, abs(err), ':', color='blue')
    ax.legend(ncol=2, prop=dict(size=12))
    ax.set_xlabel(r'$t$', fontdict={'size': 16})
    ax.set_ylabel(r'$|u(t) - V_t|$', fontdict={'size': 16})
    plt.show()

    fig.savefig(visualization_helper.get_folder() / 'sde2.pdf', bbox_inches='tight')
