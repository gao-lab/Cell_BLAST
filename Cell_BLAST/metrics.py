r"""
Functions for computing benchmark metrics
"""

import typing

import anndata as ad
import igraph
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.metrics
import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity

from . import blast, utils

_identity = lambda x, y: 1 if x == y else 0


# ===============================================================================
#
#  Cluster based metrics
#
# ===============================================================================
def confusion_matrix(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    r"""
    Reimplemented this because sklearn.metrics.confusion_matrix
    does not provide row names and column names.
    """
    x, x_c = utils.encode_integer(x)
    y, y_c = utils.encode_integer(y)
    unique_x, unique_y = np.unique(x), np.unique(y)
    cm = np.empty((len(unique_x), len(unique_y)), dtype=np.int)
    for i in unique_x:
        for j in unique_y:
            cm[i, j] = np.sum((x == i) & (y == j))
    return pd.DataFrame(data=cm, index=x_c, columns=y_c)


def class_specific_accuracy(
    true: np.ndarray, pred: np.ndarray, expectation: pd.DataFrame
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.unique(true), columns=["number", "accuracy"])
    expectation = expectation.astype(np.bool)
    for c in df.index:
        true_mask = true == c
        pred_mask = np.in1d(pred, expectation.columns[expectation.loc[c]])
        df.loc[c, "number"] = true_mask.sum()
        df.loc[c, "accuracy"] = (
            np.logical_and(pred_mask, true_mask).sum() / df.loc[c, "number"]
        )
    return df


def mean_balanced_accuracy(
    true: np.ndarray,
    pred: np.ndarray,
    expectation: pd.DataFrame,
    population_weighed: bool = False,
) -> float:
    df = class_specific_accuracy(true, pred, expectation)
    if population_weighed:
        return (df["accuracy"] * df["number"]).sum() / df["number"].sum()
    return df["accuracy"].mean()


def cl_accuracy(
    cl_dag: utils.CellTypeDAG,
    source: np.ndarray,
    target: np.ndarray,
    ref_cl_list: typing.List[str],  # a list of unique cl in ref
) -> pd.DataFrame:
    if len(source) != len(target):
        raise ValueError("Invalid input: different cell number.")
    positive_cl = []
    negative_cl = []
    for query_cl in np.unique(source):
        for ref_cl in ref_cl_list:
            if cl_dag.is_related(query_cl, ref_cl):
                positive_cl.append(query_cl)
                break
        else:
            negative_cl.append(query_cl)

    # sensitivity
    accuracy_dict = {}
    ref_cl_set = {cl_dag.get_vertex(item) for item in ref_cl_list}
    for query_cl in positive_cl:
        target_cl = target[source == query_cl]
        cl_accuracy = np.zeros(target_cl.size)
        cache = {}  # avoid repeated computation
        for i, _target_cl in enumerate(target_cl):
            if _target_cl in cache:
                cl_accuracy[i] = cache[_target_cl]
                continue
            if cl_dag.is_descendant_of(_target_cl, query_cl):
                cache[_target_cl] = 1  # equal or descendant results as 1
            elif cl_dag.is_ancestor_of(_target_cl, query_cl):
                intermediates = set.intersection(
                    set(
                        cl_dag.graph.bfsiter(
                            cl_dag.get_vertex(query_cl), mode=igraph.OUT
                        )
                    ),
                    set(
                        cl_dag.graph.bfsiter(
                            cl_dag.get_vertex(_target_cl), mode=igraph.IN
                        )
                    ),
                )
                cache[_target_cl] = np.mean(
                    [
                        len(list(cl_dag.graph.bfsiter(intermediate, mode=igraph.IN)))
                        / len(
                            list(
                                cl_dag.graph.bfsiter(
                                    cl_dag.get_vertex(_target_cl), mode=igraph.IN
                                )
                            )
                        )
                        for intermediate in intermediates
                        if ref_cl_set.intersection(
                            set(cl_dag.graph.bfsiter(intermediate, mode=igraph.IN))
                        )
                    ]
                )
            else:
                cache[_target_cl] = 0
            cl_accuracy[i] = cache[_target_cl]
        accuracy_dict[query_cl] = [cl_accuracy.size, cl_accuracy.mean(), True]

    # specificity
    for query_cl in negative_cl:
        target_cl = target[source == query_cl]
        cl_accuracy = np.in1d(target_cl, ["rejected", "unassigned"])
        accuracy_dict[query_cl] = [cl_accuracy.size, cl_accuracy.mean(), False]

    return pd.DataFrame(accuracy_dict, index=("cell number", "accuracy", "positive")).T


def cl_mba(
    cl_dag: utils.CellTypeDAG,
    source: np.ndarray,
    target: np.ndarray,
    ref_cl_list: typing.List[str],
) -> float:
    accuracy_df = cl_accuracy(
        cl_dag=cl_dag, source=source, target=target, ref_cl_list=ref_cl_list
    )
    return accuracy_df["accuracy"].mean()


# ===============================================================================
#
#  Distance based metrics
#
# ===============================================================================
def nearest_neighbor_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "minkowski",
    similarity: typing.Callable = _identity,
    n_jobs: int = 1,
) -> float:
    nearestNeighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=2, metric=metric, n_jobs=n_jobs
    )
    nearestNeighbors.fit(x)
    nni = nearestNeighbors.kneighbors(x, return_distance=False)
    return np.vectorize(similarity)(y, y[nni[:, 1].ravel()]).mean()


def mean_average_precision_from_latent(
    x: np.ndarray,
    y: np.ndarray,
    p: typing.Optional[np.ndarray] = None,
    k: float = 0.01,
    metric: str = "minkowski",
    posterior_metric: str = "npd_v1",
    similarity: typing.Callable = _identity,
    n_jobs: int = 1,
) -> float:
    if k < 1:
        k = y.shape[0] * k
    k = np.round(k).astype(np.int)
    nearestNeighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), metric=metric, n_jobs=n_jobs
    )
    nearestNeighbors.fit(x)
    nni = nearestNeighbors.kneighbors(x, return_distance=False)
    if p is not None:
        posterior_metric = getattr(blast, posterior_metric)
        pnnd = np.empty_like(nni, np.float32)
        for i in range(pnnd.shape[0]):
            for j in range(pnnd.shape[1]):
                pnnd[i, j] = posterior_metric(x[i], x[nni[i, j]], p[i], p[nni[i, j]])
            nni[i] = nni[i][np.argsort(pnnd[i])]
    return mean_average_precision(y, y[nni[:, 1:]], similarity=similarity)


def average_silhouette_score(x: np.ndarray, y: np.ndarray) -> float:
    return sklearn.metrics.silhouette_score(x, y)


def seurat_alignment_score(
    x: np.ndarray,
    y: np.ndarray,
    k: float = 0.01,
    n: int = 1,
    metric: str = "minkowski",
    random_seed: typing.Optional[int] = None,
    n_jobs: int = 1,
) -> float:
    random_state = np.random.RandomState(random_seed)
    idx_list = [np.where(y == _y)[0] for _y in np.unique(y)]
    subsample_size = min(idx.size for idx in idx_list)
    subsample_scores = []
    for _ in range(n):
        subsample_idx_list = [
            random_state.choice(idx, subsample_size, replace=False) for idx in idx_list
        ]
        subsample_y = y[np.concatenate(subsample_idx_list)]
        subsample_x = x[np.concatenate(subsample_idx_list)]
        _k = subsample_y.shape[0] * k if k < 1 else k
        _k = np.round(_k).astype(np.int)
        nearestNeighbors = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(subsample_y.shape[0], _k + 1), metric=metric, n_jobs=n_jobs
        )
        nearestNeighbors.fit(subsample_x)
        nni = nearestNeighbors.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            (subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1))
            .sum(axis=1)
            .mean()
        )
        subsample_scores.append(
            (_k - same_y_hits) * len(idx_list) / (_k * (len(idx_list) - 1))
        )
    return np.mean(subsample_scores)


def batch_mixing_entropy(
    x: np.ndarray,
    y: np.ndarray,
    boots: int = 100,
    sample_size: int = 100,
    k: int = 100,
    metric: str = "minkowski",
    random_seed: typing.Optional[int] = None,
    n_jobs: int = 1,
) -> float:
    random_state = np.random.RandomState(random_seed)
    batches = np.unique(y)
    entropy = 0
    for _ in range(boots):
        bootsamples = random_state.choice(
            np.arange(x.shape[0]), sample_size, replace=False
        )
        subsample_x = x[bootsamples]
        neighbor = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k, metric=metric, n_jobs=n_jobs
        )
        neighbor.fit(x)
        nn = neighbor.kneighbors(subsample_x, return_distance=False)
        for i in range(sample_size):
            for batch in batches:
                b = len(np.where(y[nn[i, :]] == batch)[0]) / k
                if b == 0:
                    entropy = entropy
                else:
                    entropy = entropy + b * np.log(b)
    entropy = -entropy / (boots * sample_size)
    return entropy


# ===============================================================================
#
#  Ranking based metrics
#
# ===============================================================================
def _average_precision(r: np.ndarray) -> float:
    cummean = np.cumsum(r) / (np.arange(r.size) + 1)
    mask = r > 0
    if np.any(mask):
        return cummean[mask].mean()
    return 0.0


def mean_average_precision(
    true: np.ndarray,
    hits: np.ndarray,
    similarity: typing.Callable[[typing.Any, typing.Any], float] = _identity,
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    true
        True label
    hits
        Hit labels
    similarity
        Function that defines the similarity between labels

    Returns
    -------
    map
        Mean average precision
    """
    r = np.apply_along_axis(np.vectorize(similarity), 0, hits, true)
    return np.apply_along_axis(_average_precision, 1, r).mean()


# ===============================================================================
#
#  Structure preservation
#
# ===============================================================================
def avg_neighbor_jacard(
    x: np.ndarray,
    y: np.ndarray,
    x_metric: str = "minkowski",
    y_metric: str = "minkowski",
    k: typing.Union[int, float] = 0.01,
    n_jobs: int = 1,
) -> float:
    r"""
    Average neighborhood Jacard index.

    Parameters
    ----------
    x
        First feature space.
    y
        Second feature space.
    x_metric
        Distance metric to use in first feature space.
        See :class:`sklearn.neighbors.NearestNeighbors` for available options.
    y_metric
        Distance metric to use in second feature space.
        See :class:`sklearn.neighbors.NearestNeighbors` for available options.
    k
        Number (if k is an integer greater than 1) or fraction in total data
        (if k is a float less than 1) of nearest neighbors to consider.
    n_jobs
        Number of parallel jobs to use when doing nearest neighbor search.
        See :class:`sklearn.neighbors.NearestNeighbors` for details.

    Returns
    -------
    jacard
        Average neighborhood Jacard index.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Inconsistent shape!")
    n = x.shape[0]
    k = n * k if k < 1 else k
    k = np.round(k).astype(np.int)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(n, k + 1), metric=x_metric, n_jobs=n_jobs
    ).fit(x)
    nni_x = nn.kneighbors(x, return_distance=False)[:, 1:]
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(n, k + 1), metric=y_metric, n_jobs=n_jobs
    ).fit(y)
    nni_y = nn.kneighbors(y, return_distance=False)[:, 1:]
    jacard = np.array(
        [
            np.intersect1d(i, j).size / np.union1d(i, j).size
            for i, j in zip(nni_x, nni_y)
        ]
    )
    return jacard.mean()


def jacard_index(x: scipy.sparse.csr_matrix, y: scipy.sparse.csr_matrix) -> np.ndarray:
    r"""
    Compute Jacard index between two nearest neighbor graphs

    Parameters
    ----------
    x
        First nearest neighbor graph
    y
        Second nearest neighbor graph

    Returns
    -------
    jacard
        Jacard index for each row
    """
    xy = x + y
    return (
        np.asarray((xy == 2).sum(axis=1)).ravel()
        / np.asarray((xy > 0).sum(axis=1)).ravel()
    )


def neighbor_preservation_score(
    x: np.ndarray,
    nng: scipy.sparse.spmatrix,
    metric: str = "minkowski",
    k: typing.Union[int, float] = 0.01,
    n_jobs: int = 1,
) -> float:
    if not x.shape[0] == nng.shape[0] == nng.shape[1]:
        raise ValueError("Inconsistent shape!")
    n = x.shape[0]
    k = n * k if k < 1 else k
    k = np.round(k).astype(np.int)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(n, k + 1), metric=metric, n_jobs=n_jobs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)[:, 1:]
    ap = np.array(
        [
            _average_precision(_nng.toarray().ravel()[_nni])
            for _nng, _nni in zip(nng, nni)
        ]
    )
    max_ap = np.array(
        [
            _average_precision(np.sort(_nng.toarray().ravel())[::-1][: nni.shape[1]])
            for _nng in nng
        ]
    )
    ap /= max_ap
    return ap.mean()


def calc_reference_sas(
    adata: ad.AnnData,
    batch_effect: str = "dataset_name",
    cell_ontology: str = "cell_ontology_class",
    similarity: typing.Callable = _identity,
):
    neighbors_propotion = []
    n = len(adata.obs[batch_effect].unique())
    for x in adata.obs[batch_effect].unique():
        propotion = 0
        for i in adata.obs[cell_ontology].unique():
            neighbors = 0
            own_neighbors = 0
            for j in adata.obs[cell_ontology].unique():
                own_neighbors += (
                    similarity(i, j)
                    * (
                        (adata.obs[cell_ontology] == j) & (adata.obs[batch_effect] == x)
                    ).sum()
                )
                neighbors += similarity(i, j) * (adata.obs[cell_ontology] == j).sum()

            propotion += (
                own_neighbors
                / neighbors
                * (
                    (adata.obs[cell_ontology] == i) & (adata.obs[batch_effect] == x)
                ).sum()
            )

        propotion = propotion / (adata.obs[batch_effect] == x).sum()
        neighbors_propotion.append(propotion)
    neighbors_propotion = np.mean(neighbors_propotion)
    return 1 - (neighbors_propotion - 1 / n) / (1 - 1 / n)


def mean_average_correlation(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    k: float = 0.001,
    metric: str = "minkowski",
    n_jobs: int = 1,
) -> float:
    if k < 1:
        k = y.shape[0] * k
    k = np.round(k).astype(np.int)

    nearestNeighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), metric=metric, n_jobs=n_jobs
    )
    nearestNeighbors.fit(x)
    nn = nearestNeighbors.kneighbors(x, return_distance=False)
    correlation = []
    for nni in nn:
        diff = (b != b[nni[0]])[nni]
        if diff.sum() > 0:
            correlation.append(cosine_similarity(y[nni[[0]]], y[nni][diff]).mean())
    return np.float64(np.mean(correlation))
