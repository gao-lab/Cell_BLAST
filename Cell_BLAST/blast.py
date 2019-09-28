"""
Cell BLAST based on DIRECTi models
"""

import os
import glob
import collections
import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
import sklearn.neighbors
import numba
import joblib
from . import directi
from . import metrics
from . import message
from . import data
from . import config
from . import utils

pd.options.mode.chained_assignment = None  # FIXME: this should be avoided
NORMAL = 1
MINIMAL = 0


def _wasserstein_distance_impl(x, y):  # pragma: no cover
    x_sorter = np.argsort(x)
    y_sorter = np.argsort(y)
    xy = np.concatenate((x, y))
    xy.sort()
    deltas = np.diff(xy)
    x_cdf = np.searchsorted(x[x_sorter], xy[:-1], 'right') / x.size
    y_cdf = np.searchsorted(y[y_sorter], xy[:-1], 'right') / y.size
    return np.sum(np.multiply(np.abs(x_cdf - y_cdf), deltas))


def _energy_distance_impl(x, y):  # pragma: no cover
    x_sorter = np.argsort(x)
    y_sorter = np.argsort(y)
    xy = np.concatenate((x, y))
    xy.sort()
    deltas = np.diff(xy)
    x_cdf = np.searchsorted(x[x_sorter], xy[:-1], "right") / x.size
    y_cdf = np.searchsorted(y[y_sorter], xy[:-1], "right") / y.size
    return np.sqrt(2 * np.sum(np.multiply(np.square(x_cdf - y_cdf), deltas)))


@numba.extending.overload(
    scipy.stats.wasserstein_distance, jit_options={"nogil": True, "cache": True})
def _wasserstein_distance(x, y):  # pragma: no cover
    if x == numba.float32[::1] and y == numba.float32[::1]:
        return _wasserstein_distance_impl


@numba.extending.overload(
    scipy.stats.energy_distance, jit_options={"nogil": True, "cache": True})
def _energy_distance(x, y):  # pragma: no cover
    if x == numba.float32[::1] and y == numba.float32[::1]:
        return _energy_distance_impl


@numba.jit(nopython=True, nogil=True, cache=True)
def ed(x, y):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    """
    return np.sqrt(np.square(x - y).sum())


@numba.jit(nopython=True, nogil=True, cache=True)
def _md(x, y, x_posterior):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    x_posterior = x_posterior - x
    cov_x = np.dot(x_posterior.T, x_posterior) / x_posterior.shape[0]
    dev = np.expand_dims(y - x, axis=1)
    return np.sqrt(np.dot(dev.T, np.dot(np.linalg.inv(cov_x), dev)))[0, 0]


@numba.jit(nopython=True, nogil=True, cache=True)
def md(x, y, x_posterior, y_posterior):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    y_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    return 0.5 * (_md(x, y, x_posterior) + _md(y, x, y_posterior))


@numba.jit(nopython=True, nogil=True, cache=True)
def _compute_pcasd(x, x_posterior, eps):  # pragma: no cover
    """
    x : latent_dim
    x_posterior : n_posterior * latent_dim
    """
    centered_x_posterior = x_posterior - np.sum(x_posterior, axis=0) / x_posterior.shape[0]
    cov_x = np.dot(centered_x_posterior.T, centered_x_posterior)
    v = np.real(np.linalg.eig(cov_x.astype(np.complex64))[1])  # Suppress domain change due to rounding errors
    x_posterior = np.dot(x_posterior - x, v)
    squared_x_posterior = np.square(x_posterior)
    asd = np.empty((2, x_posterior.shape[1]), dtype=np.float32)
    for p in range(x_posterior.shape[1]):
        mask = x_posterior[:, p] < 0
        asd[0, p] = np.sqrt((
            np.sum(squared_x_posterior[mask, p])
        ) / max(np.sum(mask), 1)) + eps
        asd[1, p] = np.sqrt((
            np.sum(squared_x_posterior[~mask, p])
        ) / max(np.sum(~mask), 1)) + eps
    return np.concatenate((v, asd), axis=0)


@numba.jit(nopython=True, nogil=True, cache=True)
def _compute_pcasd_across_models(x, x_posterior, eps=1e-1):  # pragma: no cover
    """
    x : n_models * latent_dim
    x_posterior : n_models * n_posterior * latent_dim
    """
    result = np.empty((x.shape[0], x.shape[-1] + 2, x.shape[-1]), dtype=np.float32)
    for i in range(x.shape[0]):
        result[i] = _compute_pcasd(x[i], x_posterior[i], eps)
    return result


@numba.jit(nopython=True, nogil=True, cache=True)
def _amd(x, y, x_posterior, eps, x_is_pcasd=False):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    if not x_is_pcasd:
        x_posterior = _compute_pcasd(x, x_posterior, eps)
    v = x_posterior[:-2]
    asd = x_posterior[-2:]
    y = np.dot(y - x, v)
    for p in range(y.size):
        if y[p] < 0:
            y[p] /= asd[0, p]
        else:
            y[p] /= asd[1, p]
    return np.linalg.norm(y)


@numba.jit(nopython=True, nogil=True, cache=True)
def amd(x, y, x_posterior, y_posterior, eps=1e-1, x_is_pcasd=False, y_is_pcasd=False):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    y_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    return 0.5 * (
        _amd(x, y, x_posterior, eps, x_is_pcasd) +
        _amd(y, x, y_posterior, eps, y_is_pcasd)
    )


@numba.jit(nopython=True, nogil=True, cache=True)
def npd_v1(x, y, x_posterior, y_posterior, eps=0.0):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    y_posterior : n_posterior * latent_dim
    """
    projection = x - y  # latent_dim
    if np.all(projection == 0):
        projection[...] = 1  # any projection is equivalent
    projection /= np.linalg.norm(projection)
    x_posterior = np.sum(x_posterior * projection, axis=1)  # n_posterior_samples
    y_posterior = np.sum(y_posterior * projection, axis=1)  # n_posterior_samples
    xy_posterior = np.concatenate((x_posterior, y_posterior))
    xy_posterior1 = (xy_posterior - np.mean(x_posterior)) / (np.std(x_posterior) + np.float32(eps))
    xy_posterior2 = (xy_posterior - np.mean(y_posterior)) / (np.std(y_posterior) + np.float32(eps))
    return 0.5 * (scipy.stats.wasserstein_distance(
        xy_posterior1[:len(x_posterior)],
        xy_posterior1[-len(y_posterior):]
    ) + scipy.stats.wasserstein_distance(
        xy_posterior2[:len(x_posterior)],
        xy_posterior2[-len(y_posterior):]
    ))


@numba.jit(nopython=True, nogil=True, cache=True)
def _npd_v2(x, y, x_posterior, eps):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    dev = y - x
    udev = dev / np.linalg.norm(dev)
    projected_noise = np.sum((x_posterior - x) * udev, axis=1)
    projected_y = np.sum((y - x) * udev)
    mask = (projected_noise * projected_y) >= 0
    scaler = np.sqrt(
        np.sum(np.square(projected_noise[mask])) /
        max(np.sum(mask), 1)
    )
    return np.abs(projected_y) / (scaler + eps)


@numba.jit(nopython=True, nogil=True, cache=True)
def npd_v2(x, y, x_posterior, y_posterior, eps=1e-1):  # pragma: no cover
    """
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    y_posterior : n_posterior * latent_dim
    """
    if np.all(x == y):
        return 0.0
    return 0.5 * (
        _npd_v2(x, y, x_posterior, eps) +
        _npd_v2(y, x, y_posterior, eps)
    )


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_ed_across_models(query_latent, ref_latent):  # pragma: no cover
    """
    query_latent : n_models * latent_dim
    ref_latent : n_hits * n_models * latent_dim
    returns : n_hits * n_models
    """
    dist = np.empty(ref_latent.shape[:-1])  # n_hits * n_models
    for i in range(dist.shape[1]):  # model index
        x = query_latent[i, ...]  # latent_dim
        for j in range(dist.shape[0]):  # hit index
            y = ref_latent[j, i, ...]  # latent_dim
            dist[j, i] = ed(x, y)
    return dist


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_md_across_models(
    query_latent, ref_latent, query_posterior, ref_posterior
):  # pragma: no cover
    """
    query_latent : n_models * latent_dim
    ref_latent : n_hits * n_models * latent_dim
    query_posterior : n_models * n_posterior * latent_dim
    ref_posterior : n_hits * n_models * n_posterior * latent_dim
    returns : n_hits * n_models
    """
    dist = np.empty(ref_latent.shape[:-1])  # n_hits * n_models
    for i in range(dist.shape[1]):  # model index
        x = query_latent[i, ...]  # latent_dim
        x_posterior = query_posterior[i, ...]  # n_posterior * latent_dim
        for j in range(dist.shape[0]):  # hit index
            y = ref_latent[j, i, ...]  # latent_dim
            y_posterior = ref_posterior[j, i, ...]  # n_posterior * latent_dim
            dist[j, i] = md(x, y, x_posterior, y_posterior)
    return dist


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_amd_across_models(
    query_latent, ref_latent, query_posterior, ref_posterior, eps=1e-1
):  # pragma: no cover
    """
    query_latent : n_models * latent_dim
    ref_latent : n_hits * n_models * latent_dim
    query_posterior : n_models * n_posterior * latent_dim
    ref_posterior : n_hits * n_models * n_posterior * latent_dim
    returns : n_hits * n_models
    """
    dist = np.empty(ref_latent.shape[:-1])  # n_hits * n_models
    for i in range(dist.shape[1]):  # model index
        x = query_latent[i, ...]  # latent_dim
        x_posterior = query_posterior[i, ...]  # n_posterior * latent_dim
        for j in range(dist.shape[0]):  # hit index
            y = ref_latent[j, i, ...]  # latent_dim
            y_posterior = ref_posterior[j, i, ...]  # n_posterior * latent_dim
            dist[j, i] = amd(
                x, y, x_posterior, y_posterior,
                eps=eps, x_is_pcasd=False, y_is_pcasd=True
            )
    return dist


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_npd_v1_across_models(
    query_latent, ref_latent, query_posterior, ref_posterior, eps=0.0
):  # pragma: no cover
    """
    query_latent : n_models * latent_dim
    ref_latent : n_hits * n_models * latent_dim
    query_posterior : n_models * n_posterior * latent_dim
    ref_posterior : n_hits * n_models * n_posterior * latent_dim
    returns : n_hits * n_models
    """
    dist = np.empty(ref_latent.shape[:-1])  # n_hits * n_models
    for i in range(dist.shape[1]):  # model index
        x = query_latent[i, ...]  # latent_dim
        x_posterior = query_posterior[i, ...]  # n_posterior * latent_dim
        for j in range(dist.shape[0]):  # hit index
            y = ref_latent[j, i, ...]  # latent_dim
            y_posterior = ref_posterior[j, i, ...]  # n_posterior * latent_dim
            dist[j, i] = npd_v1(x, y, x_posterior, y_posterior, eps=eps)
    return dist


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_npd_v2_across_models(
    query_latent, ref_latent, query_posterior, ref_posterior, eps=1e-1
):  # pragma: no cover
    """
    query_latent : n_models * latent_dim
    ref_latent : n_hits * n_models * latent_dim
    query_posterior : n_models * n_posterior * latent_dim
    ref_posterior : n_hits * n_models * n_posterior * latent_dim
    returns : n_hits * n_models
    """
    dist = np.empty(ref_latent.shape[:-1])  # n_hits * n_models
    for i in range(dist.shape[1]):  # model index
        x = query_latent[i, ...]  # latent_dim
        x_posterior = query_posterior[i, ...]  # n_posterior * latent_dim
        for j in range(dist.shape[0]):  # hit index
            y = ref_latent[j, i, ...]  # latent_dim
            y_posterior = ref_posterior[j, i, ...]  # n_posterior * latent_dim
            dist[j, i] = npd_v2(x, y, x_posterior, y_posterior, eps=eps)
    return dist


DISTANCE_METRIC_ACROSS_MODELS = {
    ed: _hit_ed_across_models,
    md: _hit_md_across_models,
    amd: _hit_amd_across_models,
    npd_v1: _hit_npd_v1_across_models,
    npd_v2: _hit_npd_v2_across_models
}


class BLAST(object):

    """
    Cell BLAST

    Parameters
    ----------
    models : list
        A list of ``Cell_BLAST.directi.DIRECTi`` models.
    ref : Cell_BLAST.data.ExprDataSet
        A reference dataset.
    distance_metric : {"npd_v1", "npd_v2", "md", "amd", "ed"}
        Cell-to-cell distance metric to use, by default "npd_v1".
    n_posterior : int
        How many samples from the posterior distribution to use for
        estimating posterior distance, by default 50. Irrelevant for
        distance_metric="ed".
    n_empirical : int
        Number of random cell pairs to use when estimating empirical
        distribution of cell-to-cell distance, by default 10000.
    cluster_empirical : bool
        Whether to build an empirical distribution for each intrinsic cluster
        independently, by default False, meaning one global empirical
        distribution is used.
    eps : float
        A small number added to the normalization factors used in certain
        posterior-based distance metrics to improve numeric stability,
        by default None, in which case a recommended value will be used
        according to the specified distance metric.
    force_components : bool
        Whether to compute all the necessary components upon initialization,
        by default True. If set to False, necessary components will be computed
        on the fly when performing queries.

    Examples
    --------

    A typical BLAST pipeline is described below.

    Assuming we have a list of ``DIRECTi`` models already fitted on some
    reference data, we can construct a BLAST object by feeding the pretrained
    models and the reference data to the ``Cell_BLAST.blast.BLAST`` constructor.

    >>> blast = BLAST(models, reference)

    We can efficiently query the reference and obtain initial hits via the ``query`` method:

    >>> hits = blast.query(query)

    Then we filter the initial hits by using more accurate metrics
    e.g. empirical p-value based on posterior distance), and pooling together
    information across multiple models.

    >>> hits = hits.reconcile_models().filter(by="pval", cutoff=0.05)

    Finally, we use the ``annotate`` method to obtain predictions based on
    reference annotations, e.g. "cell_ontology_class" in this case.

    >>> annotation = hits.annotate("cell_ontology_class")

    See the BLAST ipython notebook (:ref:`vignettes`) for live examples.
    """

    def __init__(self, models, ref, distance_metric="npd_v1", n_posterior=50,
                 n_empirical=10000, cluster_empirical=False, eps=None,
                 force_components=True, **kwargs):
        self.models = models
        self.ref = data.ExprDataSet(
            ref.exprs, ref.obs.copy(), ref.var.copy(), ref.uns.copy()
        )  # exprs and uns are shallow copied, obs and var are deep copied

        self.latent = None
        self.nearest_neighbors = None
        self.cluster = None
        self.posterior = np.array([None] * self.ref.shape[0])
        self.empirical = None

        self.distance_metric = globals()[distance_metric] \
            if isinstance(distance_metric, str) else distance_metric
        self.n_posterior = n_posterior if self.distance_metric is not ed else 0
        self.n_empirical = n_empirical
        self.cluster_empirical = cluster_empirical
        self.eps = eps

        if force_components:
            self._force_components(**kwargs)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, s):
        blast = BLAST(
            np.array(self.models)[s].tolist(),
            self.ref, self.distance_metric, self.n_posterior, self.n_empirical,
            self.cluster_empirical, self.eps, force_components=False
        )
        blast.latent = self.latent[:, s, ...] if self.latent is not None else None
        blast.cluster = self.cluster[:, s, ...] if self.cluster is not None else None
        blast.nearest_neighbors = np.array(self.nearest_neighbors)[s].tolist() \
            if self.nearest_neighbors is not None else None
        if self.posterior is not None:
            for i in range(self.posterior.size):
                if self.posterior[i] is not None:
                    blast.posterior[i] = self.posterior[i][s, ...]
        blast.empirical = [
            item for item in np.array(self.empirical)[s]
        ] if self.empirical is not None else None
        return blast

    def _get_latent(self, n_jobs):  # n_cells * n_models * latent_dim
        if self.latent is None:
            message.info("Projecting to latent space...")
            self.latent = np.stack(joblib.Parallel(
                n_jobs=min(n_jobs, len(self)), backend="threading"
            )(joblib.delayed(model.inference)(
                self.ref
            ) for model in self.models), axis=1)
        return self.latent

    def _get_cluster(self, n_jobs):  # n_cells * n_models
        if self.cluster is None:
            message.info("Obtaining intrinsic clustering...")
            self.cluster = np.stack(joblib.Parallel(
                n_jobs=min(n_jobs, len(self)), backend="threading"
            )(joblib.delayed(model.clustering)(
                self.ref
            ) for model in self.models), axis=1)
        return self.cluster

    def _get_posterior(self, n_jobs, random_seed, idx=None):  # n_cells * (n_models * n_posterior * latent_dim)
        if idx is None:
            idx = np.arange(self.ref.shape[0])
        new_idx = np.intersect1d(np.unique(idx), np.where(np.vectorize(
            lambda x: x is None
        )(self.posterior))[0])
        if new_idx.size:
            message.info("Sampling from posteriors...")
            new_ref = self.ref[new_idx, :]
            new_posterior = np.stack(joblib.Parallel(
                n_jobs=min(n_jobs, len(self)), backend="threading"
            )(joblib.delayed(model.inference)(
                new_ref, n_posterior=self.n_posterior, random_seed=random_seed
            ) for model in self.models), axis=1)  # n_cells * n_models * n_posterior * latent_dim
            # NOTE: Slow discontigous memcopy here, but that's necessary since we will be caching values by cells.
            # It also makes values more contiguous and faster to access in later cell-based operations.
            if self.distance_metric is amd:
                dist_kws = {"eps": self.eps} if self.eps is not None else {}
                new_latent = self._get_latent(n_jobs)[new_idx]
                self.posterior[new_idx] = joblib.Parallel(
                    n_jobs=n_jobs, backend="threading"
                )(joblib.delayed(_compute_pcasd_across_models)(
                    _new_latent, _new_posterior, **dist_kws
                ) for _new_latent, _new_posterior in zip(new_latent, new_posterior))
            else:
                self.posterior[new_idx] = [item for item in new_posterior]  # NOTE: No memcopy here
        return self.posterior[idx]

    def _get_nearest_neighbors(self, n_jobs):
        if self.nearest_neighbors is None:
            message.info("Fitting nearest neighbor trees...")
            latent = self._get_latent(n_jobs).swapaxes(0, 1)
            # NOTE: Makes cells discontiguous, but for nearest neighbor tree,
            # there's no influence on performance
            self.nearest_neighbors = joblib.Parallel(
                n_jobs=min(n_jobs, len(self)), backend="loky"
            )(joblib.delayed(self._fit_nearest_neighbors)(
                _latent
            ) for _latent in latent)
        return self.nearest_neighbors

    def _get_empirical(self, n_jobs, random_seed):  # n_models * [n_clusters * n_empirical]
        if self.empirical is None:
            message.info("Generating empirical null distributions...")
            if not self.cluster_empirical:
                self.cluster = np.zeros((
                    self.ref.shape[0], len(self)
                ), dtype=np.int)
            latent = self._get_latent(n_jobs)
            cluster = self._get_cluster(n_jobs)
            rs = np.random.RandomState(random_seed)
            bg = rs.choice(latent.shape[0], size=self.n_empirical)
            if self.distance_metric is not ed:
                bg_posterior = self._get_posterior(n_jobs, random_seed, idx=bg)
            self.empirical = []
            dist_kws = {"eps": self.eps} if self.eps is not None else {}
            if self.distance_metric is amd:
                dist_kws["x_is_pcasd"] = True
                dist_kws["y_is_pcasd"] = True
            for k in range(len(self)):  # model_idx
                empirical = np.zeros((np.max(cluster[:, k]) + 1, self.n_empirical))
                for c in np.unique(cluster[:, k]):  # cluster_idx
                    fg = rs.choice(np.where(cluster[:, k] == c)[0], size=self.n_empirical)
                    if self.distance_metric is ed:
                        empirical[c] = np.sort(joblib.Parallel(
                            n_jobs=n_jobs, backend="threading"
                        )(joblib.delayed(self.distance_metric)(
                            latent[fg[i]], latent[bg[i]]
                        ) for i in range(self.n_empirical)))
                    else:
                        fg_posterior = self._get_posterior(n_jobs, random_seed, idx=fg)
                        empirical[c] = np.sort(joblib.Parallel(
                            n_jobs=n_jobs, backend="threading"
                        )(joblib.delayed(self.distance_metric)(
                            latent[fg[i], k], latent[bg[i], k],
                            fg_posterior[i][k], bg_posterior[i][k],
                            **dist_kws
                        ) for i in range(self.n_empirical)))
                self.empirical.append(empirical)
        return self.empirical

    def _force_components(self, n_jobs=config._USE_GLOBAL, random_seed=config._USE_GLOBAL):
        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed
        _ = self._get_nearest_neighbors(n_jobs)
        if self.distance_metric is not ed:
            _ = self._get_posterior(n_jobs, random_seed)
        _ = self._get_empirical(n_jobs, random_seed)

    @staticmethod
    def _fit_nearest_neighbors(x):
        return sklearn.neighbors.NearestNeighbors().fit(x)

    @staticmethod
    def _nearest_neighbor_search(nn, query, n_neighbors):
        return nn.kneighbors(query, n_neighbors=n_neighbors)[1]

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _nearest_neighbor_merge(x):  # pragma: no cover
        return np.unique(x)

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _empirical_pvalue(hits, dist, cluster, empirical):  # pragma: no cover
        """
        hits : n_hits
        dist : n_hits * n_models
        cluster : n_cells * n_models
        empirical : n_models * [n_cluster * n_empirical]
        """
        pval = np.empty(dist.shape)
        for i in range(dist.shape[1]):  # model index
            for j in range(dist.shape[0]):  # hit index
                pval[j, i] = np.searchsorted(
                    empirical[i][cluster[hits[j], i]], dist[j, i]
                ) / empirical[i].shape[1]
        return pval

    def save(self, path, only_used_genes=True):
        """
        Save BLAST object to a directory.

        Parameters
        ----------
        path : str
            Specifies a path to save the BLAST object.
        only_used_genes : bool
            Whether to preserve only the genes used by models, by default True.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if self.ref is not None:
            if only_used_genes:
                if "__libsize__" not in self.ref.obs.columns:
                    self.ref.obs["__libsize__"] = np.array(
                        self.ref.exprs.sum(axis=1)
                    ).ravel()
                    # So that align will still work properly
                ref = self.ref[:, np.unique(np.concatenate([
                    model.genes for model in self.models
                ]))]
            ref.uns["distance_metric"] = self.distance_metric.__name__
            ref.uns["n_posterior"] = self.n_posterior
            ref.uns["n_empirical"] = self.n_empirical
            ref.uns["cluster_empirical"] = self.cluster_empirical
            ref.uns["eps"] = self.eps if self.eps is not None else "None"
            if self.latent is not None:
                ref.uns["latent"] = self.latent
            if self.latent is not None:
                ref.uns["cluster"] = self.cluster
            if self.empirical is not None:
                ref.uns["empirical"] = {str(i): item for i, item in enumerate(self.empirical)}
            ref.uns["posterior"] = {str(i): item for i, item in enumerate(self.posterior) if item is not None}
            ref.write_dataset(os.path.join(path, "ref.h5"))
        for i in range(len(self)):
            self.models[i].save(os.path.join(path, "model_%d" % i))

    @classmethod
    def load(cls, path, mode=NORMAL, verbose=1, **kwargs):
        """
        Load BLAST object from a directory.

        Parameters
        ----------
        path : str
            Specifies a path to load from.
        mode : {cb.blast.NORMAL, cb.blast.MINIMAL}
            If mode is set to MINIMAL, model loading will be accelerated by only
            loading the encoders, but aligning BLAST (fine-tuning) would not be
            available.
        verbose : int
            Controls model loading verbosity, by default 1.
            Check ``Cell_BLAST.model.Model`` for details.

        Returns
        -------
        blast : Cell_BLAST.blast.BLAST
            Loaded BLAST object.
        """
        assert mode in (NORMAL, MINIMAL)
        ref = data.ExprDataSet.read_dataset(os.path.join(path, "ref.h5"))
        models = []
        for model_path in sorted(glob.glob(os.path.join(path, "model_*"))):
            models.append(directi.DIRECTi.load(
                model_path, _mode=mode, verbose=verbose))
        blast = cls(
            models, ref, ref.uns["distance_metric"], ref.uns["n_posterior"],
            ref.uns["n_empirical"], ref.uns["cluster_empirical"],
            None if ref.uns["eps"] == "None" else ref.uns["eps"],
            force_components=False
        )
        blast.latent = blast.ref.uns["latent"] if "latent" in blast.ref.uns else None
        blast.cluster = blast.ref.uns["cluster"] if "latent" in blast.ref.uns else None
        blast.empirical = [
            blast.ref.uns["empirical"][str(i)] for i in range(len(blast))
        ] if "empirical" in blast.ref.uns else None
        for i in range(ref.shape[0]):
            if str(i) in blast.ref.uns["posterior"]:
                blast.posterior[i] = blast.ref.uns["posterior"][str(i)]
        blast._force_components(**kwargs)
        return blast

    def query(self, query, n_neighbors=5, n_jobs=config._USE_GLOBAL,
              random_seed=config._USE_GLOBAL):
        """
        BLAST query

        Parameters
        ----------
        query : Cell_BLAST.data.ExprDataSet
            Query transcriptomes.
        n_neighbors : int
            Initial number of nearest neighbors to search in each model,
            by default 5.
        n_jobs : int
            Number of parallel jobs to run when performing query. If not
            specified, ``Cell_BLAST.config.N_JOBS`` will be used, which defaults
            to 1. Note that each (tensorflow) job could be distributed on
            multiple CPUs for a single "job".
        random_seed : None
            Random seed for posterior sampling. If not specified,
            ``Cell_BLAST.utils.RANDOM_SEED`` will be used,
            which defaults to None.

        Returns
        -------
        hits : Cell_BLAST.blast.Hits
            Query hits
        """
        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed

        message.info("Projecting to latent space...")
        query_latent = joblib.Parallel(
            n_jobs=min(n_jobs, len(self)), backend="threading"
        )(joblib.delayed(model.inference)(
            query
        ) for model in self.models)  # n_models * [n_cells * latent_dim]

        message.info("Doing nearest neighbor search...")
        nearest_neighbors = self._get_nearest_neighbors(n_jobs)
        nni = np.stack(joblib.Parallel(
            n_jobs=min(n_jobs, len(self)), backend="threading"
        )(joblib.delayed(self._nearest_neighbor_search)(
            _nearest_neighbor, _query_latent, n_neighbors
        ) for _nearest_neighbor, _query_latent in zip(
            nearest_neighbors, query_latent
        )), axis=2)  # n_cells * n_neighbors * n_models

        message.info("Merging hits across models...")
        hits = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
            joblib.delayed(self._nearest_neighbor_merge)(_nni) for _nni in nni
        )  # n_cells * [n_hits]
        hitsu, hitsi = np.unique(np.concatenate(hits), return_inverse=True)
        hitsi = np.split(hitsi, np.cumsum([item.size for item in hits])[:-1])

        query_latent = np.stack(query_latent, axis=1)  # n_cell * n_model * latent_dim
        ref_latent = self._get_latent(n_jobs)  # n_cell * n_model * latent_dim
        if self.distance_metric is ed:
            message.info("Computing Euclidean distances...")
            dist = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_hit_ed_across_models)(
                    query_latent[i], ref_latent[hits[i]]
                ) for i in range(len(hits))
            )  # list of n_hits * n_models
        else:
            message.info("Computing posterior distribution distances...")
            query_posterior = np.stack(joblib.Parallel(
                n_jobs=min(n_jobs, len(self)), backend="threading"
            )(joblib.delayed(model.inference)(
                query, n_posterior=self.n_posterior, random_seed=random_seed
            ) for model in self.models), axis=1)  # n_cells * n_models * n_posterior_samples * latent_dim
            ref_posterior = np.stack(self._get_posterior(
                n_jobs, random_seed, idx=hitsu
            ))  # n_cells * n_models * n_posterior_samples * latent_dim
            distance_metric = DISTANCE_METRIC_ACROSS_MODELS[self.distance_metric]
            dist_kws = {"eps": self.eps} if self.eps is not None else {}
            dist = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(distance_metric)(
                    query_latent[i], ref_latent[hits[i]],
                    query_posterior[i], ref_posterior[hitsi[i]], **dist_kws
                ) for i in range(len(hits))
            )  # list of n_hits * n_models

        message.info("Computing empirical p-values...")
        empirical = self._get_empirical(n_jobs, random_seed)
        cluster = self._get_cluster(n_jobs)
        pval = joblib.Parallel(
            n_jobs=n_jobs, backend="threading"
        )(joblib.delayed(self._empirical_pvalue)(
            _hits, _dist, cluster, empirical
        ) for _hits, _dist in zip(hits, dist))  # list of n_hits * n_models

        return Hits(self.ref, hits, dist, pval, query.obs_names)

    def align(self, query, n_jobs=config._USE_GLOBAL,
              random_seed=config._USE_GLOBAL, path=".", **kwargs):
        """
        Align internal DIRECTi models with query datasets (fine tuning).

        Parameters
        ----------
        query : Cell_BLAST.data.ExprDataSet, dict
            A query dataset or a dict of query datasets, which will be aligned
            to the reference.
        n_jobs : int
            Number of parallel jobs to run when building the BLAST index,
            If not specified, ``Cell_BLAST.config.N_JOBS`` will be used,
            which defaults to 1. Note that each (tensorflow) job could be
            distributed on multiple CPUs for a single "job".
        random_seed : int
            Random seed for posterior sampling. If not specified,
            ``Cell_BLAST.config.RANDOM_SEED`` will be used,
            which defaults to None.
        path : str
            Specifies a path to store temporary files, by default ".",
            i.e. the current directory.
        **kwargs
            Additional keyword parameters passed to
            ``Cell_BLAST.directi.align_DIRECTi``.

        Returns
        -------
        blast : Cell_BLAST.blast.BLAST
            A new BLAST object with aligned internal models.
        """
        if any(model._mode == directi._TEST for model in self.models):  # pragma: no cover
            raise Exception("Align not available!")
        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed

        aligned_models = joblib.Parallel(
            n_jobs=n_jobs, backend="threading"
        )(
            joblib.delayed(directi.align_DIRECTi)(
                self.models[i], self.ref, query, random_seed=random_seed,
                path=os.path.join(path, "aligned_model_%d" % i), **kwargs
            ) for i in range(len(self))
        )
        return BLAST(
            aligned_models, self.ref, distance_metric=self.distance_metric,
            n_posterior=self.n_posterior, n_empirical=self.n_empirical,
            cluster_empirical=self.cluster_empirical, eps=self.eps
        )


class Hits(object):

    """
    BLAST hits

    Parameters
    ----------
    ref : Cell_BLAST.data.ExprDataSet
        The reference dataset.
    hits : list
        Indices of hit cell in the reference dataset.
        Each list element contains hit cell indices for a query cell.
    dist : list
        Hit cell distances.
        Each list element contains distances for a query cell.
        Each list element is a :math:`n\\_hits \\times n\\_models` matrix,
        with matrix entries corresponding to the distance to
        each hit cell under each model.
    pval : list
        Hit cell empirical p-values.
        Each list element contains p-values for a query cell.
        Each list element is a :math:`n\\_hits \\times n\\_models` matrix,
        with matrix entries corresponding to the empirical p-value of
        each hit cell under each model.
    names : array_like
        Query cell names.
    """

    FILTER_BY_DIST = 0
    FILTER_BY_PVAL = 1

    def __init__(self, ref, hits, dist, pval, names):
        self.ref = ref
        self.hits = np.array(hits)
        self.dist = np.array(dist)
        self.pval = np.array(pval)
        self.names = np.array(names)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for _hits, _dist, _pval, _name in zip(self.hits, self.dist, self.pval, self.names):
            yield Hits(self.ref, [_hits], [_dist], [_pval], [_name])

    def __getitem__(self, s):
        return Hits(self.ref, self.hits[s], self.dist[s], self.pval[s], self.names[s])

    def to_data_frames(self):
        """
        Construct hit data frames for query cells.
        Note that only reconciled ``Hits`` objects are supported.

        Returns
        -------
        data_frame_dicts : list
            Each element is hit data frame for a cell
        """
        assert self.dist[0].ndim == 1 and self.pval[0].ndim == 1  # "reconcile_models" has been called
        df_dict = collections.OrderedDict()
        for i, name in enumerate(self.names):
            df_dict[name] = self.ref.obs.iloc[self.hits[i], :]
            df_dict[name]["hits"] = self.hits[i]
            df_dict[name]["dist"] = self.dist[i]
            df_dict[name]["pval"] = self.pval[i]
        return df_dict

    def reconcile_models(self, dist_method="mean", pval_method="gmean"):
        """
        Integrate model-specific distances and empirical p-values.

        Parameters
        ----------
        dist_method : {"mean", "gmean", "min", "max"}
            Specifies how to integrate distances across difference models,
            by default "mean".
        pval_method : {"mean", "gmean", "min", "max"}
            Specifies how to integrate empirical p-values across
            different models, by default "gmean".

        Returns
        -------
        reconciled_hits : Cell_BLAST.blast.Hits
            Hit object containing reconciled
        """
        dist_method = self._get_reconcile_method(dist_method)
        dist = [dist_method(item, axis=1) for item in self.dist]
        pval_method = self._get_reconcile_method(pval_method)
        pval = [pval_method(item, axis=1) for item in self.pval]
        return Hits(self.ref, self.hits, dist, pval, self.names)

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _filter_hits(hits, dist, pval, by, cutoff, model_tolerance):  # pragma: no cover
        """
        hits : n_hits
        dist : n_hits * n_models
        pval : n_hits * n_models
        """
        if by == 0:  # Hits.FILTER_BY_DIST
            hit_mask = (dist.shape[1] - (dist <= cutoff).sum(axis=1)) <= model_tolerance
        else:  # Hits.FILTER_BY_PVAL
            hit_mask = (pval.shape[1] - (pval <= cutoff).sum(axis=1)) <= model_tolerance
        return hits[hit_mask], dist[hit_mask], pval[hit_mask]

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def _filter_reconciled_hits(hits, dist, pval, by, cutoff):  # pragma: no cover
        """
        hits : n_hits
        dist : n_hits
        pval : n_hits
        """
        if by == 0:  # Hits.FILTER_BY_DIST
            hit_mask = dist <= cutoff
        else:  # Hits.FILTER_BY_PVAL
            hit_mask = pval <= cutoff
        return hits[hit_mask], dist[hit_mask], pval[hit_mask]

    def filter(self, by="pval", cutoff=0.05, model_tolerance=0, n_jobs=1):
        """
        Filter hits by posterior distance or p-value

        Parameters
        ----------
        by : {"dist", "pval"}
            Specifies a metric based on which to filter hits, by default "pval".
        cutoff : float
            Cutoff when filtering hits, by default 0.05.
        model_tolerance : int
            Maximal number of models allowed in which the cutoff is not
            satisfied, above which the query cell will be rejected,
            by default 0. Irrelevant for reconciled hits.
        n_jobs : int
            Number of parallel jobs to run, by default 1.

        Returns
        -------
        filtered_hits : Cell_BLAST.blast.Hits
            Hit object containing remaining hits after filtering
        """
        if by == "pval":
            assert self.pval is not None
            by = Hits.FILTER_BY_PVAL
        else:  # by == "dist"
            by = Hits.FILTER_BY_DIST
        if self.dist[0].ndim == 1:
            hits, dist, pval = [_ for _ in zip(*joblib.Parallel(
                n_jobs=n_jobs, backend="threading"
            )(joblib.delayed(self._filter_reconciled_hits)(
                _hits, _dist, _pval, by, cutoff
            ) for _hits, _dist, _pval in zip(self.hits, self.dist, self.pval)))]
        else:
            hits, dist, pval = [_ for _ in zip(*joblib.Parallel(
                n_jobs=n_jobs, backend="threading"
            )(joblib.delayed(self._filter_hits)(
                _hits, _dist, _pval, by, cutoff, model_tolerance
            ) for _hits, _dist, _pval in zip(self.hits, self.dist, self.pval)))]
        return Hits(self.ref, hits, dist, pval, self.names)

    def annotate(self, field, min_hits=2, majority_threshold=0.5, return_evidence=False):
        """
        Annotate query cells based on existing annotations of hit cells.
        Fields in the meta data or gene expression values can all be specified
        for annotation / prediction. Note that for gene expression, predicted
        values are in log-scale, and user should do proper normalization
        of the reference data in advance.

        Parameters
        ----------
        field : str
            Specifies a meta column or gene name to use for annotation.
        min_hits : int
            Minimal number of hits required for annotating a query cell,
            otherwise the query cell will be rejected, by default 2.
        majority_threshold : float
            Minimal majority fraction (not inclusive) required for confident
            annotation, by default 0.5. Only effective when predicting
            categorical variables. If the threshold is not met, annotation
            will be "ambiguous".
        return_evidence : bool
            Whether to return evidence level of the annotations.

        Returns
        -------
        prediction : pandas.DataFrame
            Each row contains the inferred annotation for a query cell.
            If ``return_evidence`` is set to False, the data frame contains only
            one column, i.e. the inferred annotation.
            If ``return_evidence`` is set to True, the data frame also contains
            the number of hits, as well as the majority fraction (only for
            categorical annotations) for each query cell.
        """
        ref = self.ref.get_meta_or_var([field], normalize_var=False, log_var=True).values.ravel()
        n_hits = np.repeat(0, len(self.hits))
        if np.issubdtype(ref.dtype.type, np.character) or np.issubdtype(ref.dtype.type, np.object_):
            prediction = np.repeat("rejected", len(self.hits)).astype(object)
            majority_frac = np.repeat(np.nan, len(self.hits))
            for i, _hits in enumerate(self.hits):
                n_hits[i] = len(_hits)
                if n_hits[i] < min_hits:
                    continue
                hits = ref[_hits]
                label, count = np.unique(hits, return_counts=True)
                best_idx = count.argmax()
                majority_frac[i] = count[best_idx] / len(hits)
                if majority_frac[i] <= majority_threshold:
                    prediction[i] = "ambiguous"
                    continue
                prediction[i] = label[best_idx]
            prediction = prediction.astype(ref.dtype.type)
        elif np.issubdtype(ref.dtype.type, np.number):
            prediction = np.repeat(np.nan, len(self.hits))
            for i, _hits in enumerate(self.hits):
                n_hits[i] = len(_hits)
                if n_hits[i] < min_hits:
                    continue
                prediction[i] = ref[_hits].mean()
                # np.array call is for 1-d mean that produces 0-d values
            prediction = np.stack(prediction, axis=0)
        else:  # pragma: no cover
            raise ValueError("Unsupported data type!")
        result = collections.OrderedDict()
        result[field] = prediction
        if return_evidence:
            result["n_hits"] = n_hits
            if "majority_frac" in locals():
                result["majority_frac"] = majority_frac
        return pd.DataFrame(result, index=self.names)

    @staticmethod
    def _get_reconcile_method(method):
        if method == "mean":
            return np.mean
        if method == "gmean":
            return scipy.stats.gmean
        if method == "min":
            return np.min
        if method == "max":
            return np.max
        raise ValueError("Unknown method!")  # pragma: no cover


def sankey(query, ref, title="Sankey", width=500, height=500, tint_cutoff=1,
           font="Arial", font_size=10, suppress_plot=False):  # pragma: no cover
    """
    Make a sankey diagram of query-reference mapping (only works in
    ipython notebooks).

    Parameters
    ----------
    query : array_like
        1-dimensional array of query annotation.
    ref : array_like
        1-dimensional array of BLAST prediction based on reference database.
    title : str
        Diagram title, by default "Sankey".
    width : int
        Graph width, by default 500.
    height : int
        Graph height, by default 500.
    tint_cutoff : int
        Cutoff below which sankey flows are shown in a tinter color,
        by default 1.
    font : str
        Font family used for the plot, by default "Arial".
    font_size : float
        Font size for the plot, by default 10.
    suppress_plot : bool
        Whether to suppress plotting and only return the figure dict,
        by default False.

    Returns
    -------
    fig : dict
        Figure object fed to `iplot` of the `plotly` module to produce the plot.
    """
    cf = metrics.confusion_matrix(query, ref)
    cf["query"] = cf.index.values
    cf = cf.melt(
        id_vars=["query"], var_name="reference", value_name="count"
    ).sort_values(by="count")
    query_i, query_c = utils.encode_integer(cf["query"])
    ref_i, ref_c = utils.encode_integer(cf["reference"])

    sankey_data = dict(
        type="sankey",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5
            ),
            label=np.concatenate([
                query_c, ref_c
            ], axis=0),
            color=["#E64B35"] * len(query_c) +
                  ["#4EBBD5"] * len(ref_c)
        ),
        link=dict(
            source=query_i.tolist(),
            target=(
                ref_i + len(query_c)
            ).tolist(),
            value=cf["count"].tolist(),
            color=np.vectorize(
                lambda x: "#F0F0F0" if x <= tint_cutoff else "#CCCCCC"
            )(cf["count"])
        )
    )
    sankey_layout = dict(
        title=title,
        width=width,
        height=height,
        font=dict(
            family=font,
            size=font_size
        )
    )

    fig = dict(data=[sankey_data], layout=sankey_layout)
    if not suppress_plot:
        import plotly.offline
        plotly.offline.init_notebook_mode()
        plotly.offline.iplot(fig, validate=False)
    return fig
