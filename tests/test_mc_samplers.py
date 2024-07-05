import jax
import time
import jax.numpy as jnp
import pytest
from mlmc import NormalShiftMC

@pytest.mark.parametrize("shift", [-0.5, -0.1, 0.0, 0.1, 0.5])
def test_normal_shift(shift):
    dimension = 2
    num_traj = 200
    step_size = 100
    N_steps = 10

    rng = jax.random.key(int(time.time()))

    shift = shift + jnp.zeros((dimension,))
    model = NormalShiftMC(shift=shift, k = 0.)

    initial_conditions = jnp.zeros((num_traj, dimension))
    trajectory = [initial_conditions]
    for i in range(N_steps):
        rng, rng_use = jax.random.split(rng)
        positions, p_acc = model(trajectory[-1], rng_use, step_size)
        trajectory.append(positions)
        print(i/N_steps, jnp.mean(positions, axis=0), jnp.var(positions, axis=0), p_acc)

    trajectory = jnp.asarray(trajectory)

    absolute_average = jnp.mean(trajectory)
    # We expect a zero mean of the trajectories, independent of shift in the distribution
    assert absolute_average < 1
