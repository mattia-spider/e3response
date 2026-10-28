import jraph
import numpy as np
import pytest

from e3response.stats import ResidualMeanSquare

rng = np.random.default_rng(0)


def _batch(n_node, z, born, diel, energy, pad_nodes=0, pad_graphs=0):
    """A batch with node/graph padding (mask False on the padding)."""
    n_real = sum(n_node)
    node_mask = np.r_[np.ones(n_real, bool), np.zeros(pad_nodes, bool)]
    graph_mask = np.r_[np.ones(len(n_node), bool), np.zeros(pad_graphs, bool)]
    pad = lambda a, k: np.concatenate([a, rng.normal(size=(k,) + a.shape[1:]) * 50])  # noqa: E731
    return jraph.GraphsTuple(
        n_node=np.r_[n_node, [pad_nodes] + [0] * (pad_graphs - 1)] if pad_graphs else np.asarray(n_node),
        n_edge=np.zeros(len(n_node) + pad_graphs, int),
        nodes={"atomic_numbers": np.r_[z, np.zeros(pad_nodes, int)], "born_charges": pad(born, pad_nodes),
               "mask": node_mask},
        edges={},
        globals={"dielectric_tensor": pad(diel, pad_graphs), "energy": pad(energy, pad_graphs),
                 "mask": graph_mask},
        senders=np.zeros(0, int),
        receivers=np.zeros(0, int),
    )


def _data(n_graphs):
    n_node = rng.integers(2, 6, n_graphs).tolist()
    z = rng.choice([8, 22, 56], size=sum(n_node))
    born = rng.normal(size=(len(z), 3, 3)) + (z[:, None, None] / 10.0) * np.eye(3)
    diel = rng.normal(size=(n_graphs, 3, 3)) + 6.0 * np.eye(3)
    graph_of = np.repeat(np.arange(n_graphs), n_node)
    per_z = {8: -430.0, 22: -1600.0, 56: -700.0}
    energy = np.bincount(graph_of, weights=[per_z[k] for k in z]) + rng.normal(size=n_graphs) * 0.3
    return n_node, z, born, diel, energy


def _accumulate(metric, batches):
    out = metric.create(batches[0])
    for b in batches[1:]:
        out = out.merge(metric.create(b))
    return out.compute()


def test_residual_mean_square_baselines():
    data = [_data(4), _data(3)]
    batches = [_batch(*d, pad_nodes=3, pad_graphs=2) for d in data]
    z = np.concatenate([d[1] for d in data])
    born = np.concatenate([d[2] for d in data])
    diel = np.concatenate([d[3] for d in data])

    # zero baseline: plain mean square over components
    ref = (born**2).mean()
    assert np.isclose(_accumulate(ResidualMeanSquare("nodes.born_charges"), batches), ref)
    assert np.isclose(
        _accumulate(ResidualMeanSquare("nodes.born_charges", inverse=True), batches), 1 / ref
    )

    # mean baseline: spread about the mean tensor
    ref = ((diel - diel.mean(axis=0)) ** 2).mean()
    assert np.isclose(_accumulate(ResidualMeanSquare("globals.dielectric_tensor", "mean"), batches), ref)

    # isotropic baseline on a global tensor
    iso = np.trace(diel, axis1=1, axis2=2).mean() / 3
    ref = ((diel - iso * np.eye(3)) ** 2).mean()
    got = _accumulate(ResidualMeanSquare("globals.dielectric_tensor", "isotropic"), batches)
    assert np.isclose(got, ref)

    # per-species isotropic baseline on a per-atom tensor
    res = born.copy()
    for k in np.unique(z):
        res[z == k] -= np.trace(born[z == k], axis1=1, axis2=2).mean() / 3 * np.eye(3)
    got = _accumulate(ResidualMeanSquare("nodes.born_charges", "species_isotropic"), batches)
    assert np.isclose(got, (res**2).mean())


def test_residual_mean_square_species_lstsq():
    data = [_data(20), _data(15)]
    batches = [_batch(*d, pad_nodes=2, pad_graphs=1) for d in data]
    counts, energy = [], []
    for n_node, z, _, _, e in data:
        graph_of = np.repeat(np.arange(len(n_node)), n_node)
        counts.append(np.stack([np.bincount(graph_of[z == k], minlength=len(n_node)) for k in (8, 22, 56)], 1))
        energy.append(e)
    a, y = np.concatenate(counts).astype(float), np.concatenate(energy)
    ref = ((y - a @ np.linalg.lstsq(a, y, rcond=None)[0]) ** 2).mean()
    got = _accumulate(ResidualMeanSquare("globals.energy", "species_lstsq"), batches)
    assert np.isclose(got, ref, rtol=1e-6)
    assert got < 1.0  # the per-element sum explains the -1e3 eV energies, leaving the noise


def test_residual_mean_square_missing_field_and_bad_baseline():
    import reax

    batch = _batch(*_data(2))
    with pytest.raises(reax.exceptions.DataNotFound):
        ResidualMeanSquare("nodes.raman_tensors").create(batch)
    with pytest.raises(ValueError):
        ResidualMeanSquare("globals.dielectric_tensor", "species_isotropic")


def test_species_lstsq_isotropic_and_contributions():
    """A global tensor whose isotropic part is a sum of per-element terms: the residual is what is
    left after that sum (with an anisotropic part untouched), the coefficients are the per-element
    terms, and a batch of padding only adds nothing."""
    from e3response.stats import SpeciesIsotropicContributions

    data = [_data(25), _data(20)]
    per_z = {8: -3.0, 22: -30.0, 56: 6.0}
    chi, counts, batches = [], [], []
    for n_node, z, born, diel, energy in data:
        graph_of = np.repeat(np.arange(len(n_node)), n_node)
        iso = np.bincount(graph_of, weights=[per_z[k] for k in z])
        chi.append(iso[:, None, None] * np.eye(3) + rng.normal(size=(len(n_node), 3, 3)))
        counts.append(np.stack([np.bincount(graph_of[z == k], minlength=len(n_node)) for k in (8, 22, 56)], 1))
        b = _batch(n_node, z, born, diel, energy, pad_nodes=2, pad_graphs=1)
        b.globals["chi"] = np.concatenate([chi[-1], rng.normal(size=(1, 3, 3)) * 50])
        batches.append(b)
    padding_only = _batch(*data[0], pad_nodes=2, pad_graphs=1)
    padding_only.globals["chi"] = rng.normal(size=(len(data[0][0]) + 1, 3, 3)) * 50
    padding_only.globals["mask"] = np.zeros(len(data[0][0]) + 1, bool)
    batches.append(padding_only)

    a_mat, c_all = np.concatenate(counts).astype(float), np.concatenate(chi)
    iso = np.trace(c_all, axis1=1, axis2=2) / 3
    coeffs = np.linalg.lstsq(a_mat, iso, rcond=None)[0]
    ref = ((c_all - (a_mat @ coeffs)[:, None, None] * np.eye(3)) ** 2).mean()

    got = _accumulate(ResidualMeanSquare("globals.chi", "species_lstsq_isotropic"), batches)
    assert np.isclose(got, ref, rtol=1e-6)
    contributions = _accumulate(
        SpeciesIsotropicContributions("globals.chi", type_map=[8, 22, 56, 1], divide_by=2.0), batches)
    np.testing.assert_allclose(contributions[:3], coeffs / 2.0, rtol=1e-6)
    np.testing.assert_allclose(contributions[:3], [-1.5, -15.0, 3.0], atol=0.2)
    assert contributions[3] == 0.0  # absent element


def test_species_contributions_float64():
    """Per-element energies of -1e3 eV cells, in the shape per_species_rescale takes as shifts, with
    the residual of the least-squares fit (also when one element is absent)."""
    from e3response.stats import SpeciesContributions

    data = [_data(20), _data(15)]
    batches = [_batch(*d, pad_nodes=2, pad_graphs=1) for d in data]
    counts, energy = [], []
    for n_node, z, _, _, e in data:
        graph_of = np.repeat(np.arange(len(n_node)), n_node)
        counts.append(np.stack([np.bincount(graph_of[z == k], minlength=len(n_node)) for k in (8, 22, 56)], 1))
        energy.append(e)
    a, y = np.concatenate(counts).astype(float), np.concatenate(energy)
    shifts = _accumulate(SpeciesContributions("globals.energy", type_map=[8, 22, 56, 1]), batches)
    assert np.shape(shifts) == (4, 1) and shifts[3] == [0.0]
    np.testing.assert_allclose(np.ravel(shifts)[:3], np.linalg.lstsq(a, y, rcond=None)[0], rtol=1e-9)
    np.testing.assert_allclose(np.ravel(shifts)[:3], [-430.0, -1600.0, -700.0], atol=1.0)
