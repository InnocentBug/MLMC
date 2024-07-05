from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory):
    data = np.asarray(trajectory)

    print(data.shape)

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for i in range(data.shape[1]):
        ax.plot(data[:, i, 0], data[:, i, 1])

    fig.savefig("traj.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@dataclass
class NormalShiftMC:
    shift: jnp.ndarray | None
    k: float = 0.0

    def energy(self, x):
        return self.k / 2.0 * jnp.sum(x**2, axis=1)

    def __call__(self, x, rng, N):
        shift = self.shift
        if shift is None:
            shift = jnp.zeros(x.shape[1])

        def inner(_, val):
            x, n_acc, rng = val
            rng, rng_distr, rng_acc = jax.random.split(rng, 3)

            y = jax.random.normal(rng_distr, shape=x.shape)
            y += shift[None, :]

            new_x = x + y
            old_energy = self.energy(x)
            new_energy = self.energy(new_x)

            p_energy = jnp.exp(old_energy - new_energy)
            p_forward = jnp.exp(-jnp.sum((y - shift[None, :]) ** 2, axis=1) / 2)
            p_backward = jnp.exp(-jnp.sum((y + shift[None, :]) ** 2, axis=1) / 2)

            p_accept = p_energy * p_backward / p_forward
            rand = jax.random.uniform(rng_acc, (x.shape[0],))
            acc = rand <= p_accept

            n_acc += jnp.sum(acc, dtype=int)

            x = x + acc[:, None] * y
            return x, n_acc, rng

        x, n_acc, _ = jax.lax.fori_loop(0, N, inner, (x, 0, rng))
        p_acc = n_acc / (x.shape[0] * N)
        return x, p_acc
