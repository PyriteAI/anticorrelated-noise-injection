from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from optax import GradientTransformation


class AddNoiseState(NamedTuple):
    noise: chex.ArrayTree
    rng_key: chex.PRNGKey


def add_anticorrelated_noise(eta: float, rng_key: chex.PRNGKey) -> GradientTransformation:
    stddev = jnp.sqrt(eta)

    def init_fn(params):
        # Generate an initial noise PyTree
        num_vars = len(jax.tree_leaves(params))
        treedef = jax.tree_structure(params)

        all_keys = jax.random.split(rng_key, num=num_vars + 1)
        noise = jax.tree_multimap(
            lambda p, k: jax.random.normal(k, shape=p.shape, dtype=p.dtype),
            params,
            jax.tree_unflatten(treedef, all_keys[1:]),
        )

        return AddNoiseState(noise=noise, rng_key=all_keys[0])

    def update_fn(updates, state, params=None):
        del params

        num_vars = len(jax.tree_leaves(updates))
        treedef = jax.tree_structure(updates)

        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_multimap(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype) * stddev,
            updates,
            jax.tree_unflatten(treedef, all_keys[1:]),
        )
        diff = jax.tree_multimap(lambda n, n_tm1: n - n_tm1, noise, state.noise)
        updates = jax.tree_multimap(lambda g, d: g + d * stddev, updates, diff)

        return updates, AddNoiseState(noise=noise, rng_key=all_keys[0])

    return GradientTransformation(init_fn, update_fn)


__all__ = ["add_anticorrelated_noise"]
