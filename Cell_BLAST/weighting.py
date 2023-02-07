r"""
Weighting strategy for adversarial batch alignment in DIRECTi
"""

import time
import typing
from collections import Counter

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.pyplot import rc_context
from sklearn.metrics.pairwise import pairwise_distances

from . import config, data, utils

_identity = lambda x, y: 1 if x == y else 0


def calc_weights(
    adata: ad.AnnData,
    genes: typing.Optional[typing.List[str]],
    batch_effect: typing.Optional[str],
    add_weight: typing.Tuple[bool],
    clustering_space: str,
    similarity_space: str,
    random_seed: int,
) -> None:
    r"""
    Calculate the proper weight of each cell for adversarial batch alignment.

    Parameters
    ----------
    adata
        Dataset to be calucated.
    batch_effect
        Batch effects need to be corrected.
    weighting
        Whether to use the weighting strategy or just return the default weight.
    clustering_space
        The name of the space used for clustering
    similiarity_space
        The name of the space used for similarity calculating
    random_seed
        Random seed. If not specified, :data:`config.RANDOM_SEED`
        will be used, which defaults to 0.

    Returns
    -------
    data_dict
        A :class:`utils.Datadict` object with weight information added
    """

    if any(add_weight):
        utils.logger.info("Calculating weights...")
    start_time = time.time()

    for _add_weight, _batch_effect in zip(add_weight, batch_effect):
        if _add_weight:
            if config.SUPERVISION is None:
                if clustering_space is None:
                    clustering_latent = get_default_clustering_space(
                        adata, genes, _batch_effect, random_seed
                    )
                else:
                    clustering_latent = adata.obsm[clustering_space]

                if similarity_space is None:
                    similarity_latent = get_default_similarity_space(
                        adata, genes, _batch_effect, random_seed
                    )
                else:
                    similarity_latent = adata.obsm[similarity_space]

            weight_full = np.zeros(adata.n_obs, dtype=np.float32)

            batch = utils.densify(
                utils.encode_onehot(adata.obs[_batch_effect], sort=True)
            )
            num_batch = batch.shape[1]
            mask = batch.sum(axis=1) > 0
            batch = batch[mask].argmax(axis=1)
            adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_mask"] = mask
            adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_batch"] = batch
            adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_num_batch"] = num_batch

            if config.SUPERVISION is not None:
                all_labels = adata.obs[config.SUPERVISION][mask]

                cluster, sum_cluster, num_clusters = get_supervised_cluster(
                    all_labels, batch, num_batch
                )
                adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_cluster"] = cluster
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_sum_cluster"
                ] = sum_cluster
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_num_clusters"
                ] = num_clusters

                volume = get_volume(
                    batch, num_batch, cluster, sum_cluster, num_clusters
                )
                adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_volume"] = volume

                similarity = get_supervised_similarity(all_labels, cluster, sum_cluster)
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_raw_similarity"
                ] = similarity

            else:  # supervision is None
                cluster, sum_cluster, num_clusters = get_unsupervised_cluster(
                    clustering_latent[mask], batch, num_batch, random_seed
                )
                adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_cluster"] = cluster
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_sum_cluster"
                ] = sum_cluster
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_num_clusters"
                ] = num_clusters

                volume = get_volume(
                    batch, num_batch, cluster, sum_cluster, num_clusters
                )
                adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_volume"] = volume

                similarity, raw_similarity = get_unsupervised_similarity(
                    similarity_latent[mask],
                    num_batch,
                    cluster,
                    sum_cluster,
                    num_clusters,
                    volume,
                )
                adata.uns[
                    config._WEIGHT_PREFIX_ + _batch_effect + "_raw_similarity"
                ] = raw_similarity

            weight = get_weight(
                batch, num_batch, cluster, sum_cluster, num_clusters, volume, similarity
            )
            adata.uns[config._WEIGHT_PREFIX_ + _batch_effect + "_weight"] = weight

            weight_full[mask] = weight

            report = f'[batch effect "{_batch_effect}"] '
            report += f"time elapsed={time.time() - start_time:.1f}s"
            print(report)

        else:
            weight_full = np.ones(adata.n_obs, dtype=np.float32)

        adata.obs[config._WEIGHT_PREFIX_ + _batch_effect] = weight_full


def get_supervised_cluster(
    all_labels: typing.List[str], batch: np.ndarray, num_batch: int
) -> typing.Tuple[np.ndarray, int, typing.List[int], np.ndarray]:
    sum_cluster = 0
    num_clusters = [0]
    cluster = np.zeros(batch.shape[0], dtype=np.int)

    for i in range(num_batch):
        if config.NO_CLUSTER:
            cluster[batch == i] = np.array(list(range((batch == i).sum())))
            cluster[batch == i] += sum_cluster
            sum_cluster = cluster[batch == i].max() + 1
            num_clusters.append(sum_cluster)
        else:
            unique_labels = pd.Series(all_labels[batch == i]).unique()
            for j, label in enumerate(unique_labels):
                cluster[(batch == i) & (all_labels == label)] = j
            cluster[batch == i] += sum_cluster
            sum_cluster = cluster[batch == i].max() + 1
            num_clusters.append(sum_cluster)

    return cluster, sum_cluster, num_clusters


def get_supervised_similarity(
    all_labels: typing.List[str], cluster: np.ndarray, sum_cluster: int
) -> np.ndarray:
    similarity = np.zeros((sum_cluster, sum_cluster))
    for i in range(sum_cluster):
        for j in range(sum_cluster):
            if list(all_labels[cluster == i])[0] == list(all_labels[cluster == j])[0]:
                similarity[i, j] = 1

    return similarity


def get_unsupervised_cluster(
    clustering_latent: np.ndarray, batch: np.ndarray, num_batch: int, random_seed: int
) -> typing.Tuple[np.ndarray, int, typing.List[int], np.ndarray]:
    sum_cluster = 0
    num_clusters = [0]
    cluster = np.zeros(batch.shape[0], dtype=np.int)

    for i in range(num_batch):
        if config.NO_CLUSTER:
            cluster[batch == i] = np.array(list(range((batch == i).sum())))
            cluster[batch == i] += sum_cluster
            sum_cluster = cluster[batch == i].max() + 1
            num_clusters.append(sum_cluster)
        else:
            curr_latent = ad.AnnData(clustering_latent[batch == i])
            sc.pp.neighbors(curr_latent, use_rep="X", random_state=random_seed)
            sc.tl.leiden(
                curr_latent, random_state=random_seed, resolution=config.RESOLUTION
            )
            cluster[batch == i] = np.asarray(curr_latent.obs["leiden"].astype(int))
            cluster[batch == i] += sum_cluster
            sum_cluster = cluster[batch == i].max() + 1
            num_clusters.append(sum_cluster)

    return cluster, sum_cluster, num_clusters


def get_volume(
    batch: np.ndarray,
    num_batch: int,
    cluster: np.ndarray,
    sum_cluster: int,
    num_clusters: typing.List[int],
):
    volume = np.zeros(sum_cluster)
    for i in range(num_batch):
        for j in range(num_clusters[i], num_clusters[i + 1]):
            volume[j] = (cluster == j).sum() / (batch == i).sum()

    return volume


def get_unsupervised_similarity(
    similarity_latent: np.ndarray,
    num_batch: int,
    cluster: np.ndarray,
    sum_cluster: int,
    num_clusters: typing.List[int],
    volume: np.ndarray,
) -> typing.Tuple[np.ndarray]:
    center = np.zeros((sum_cluster, similarity_latent.shape[1]))
    for i in range(sum_cluster):
        center[i] = similarity_latent[cluster == i].mean(axis=0)

    raw_similarity = -pairwise_distances(
        center, metric=config.METRIC, **config.METRIC_KWARGS
    )

    if config.MNN:
        raw_similarity = get_MNN(
            num_batch, cluster, sum_cluster, num_clusters, raw_similarity
        )
    else:
        raw_similarity = (raw_similarity - config.THRESHOLD) / (1 - config.THRESHOLD)
        raw_similarity[raw_similarity < 0] = 0

    similarity = raw_similarity.copy()
    for i in range(sum_cluster):
        similarity[:, i] *= volume
        for j in range(num_batch):
            curr_sum = similarity[num_clusters[j] : num_clusters[j + 1], i].sum()
            if curr_sum > 0:
                similarity[num_clusters[j] : num_clusters[j + 1], i] /= curr_sum

    return similarity, raw_similarity


def get_weight(
    batch: np.ndarray,
    num_batch: int,
    cluster: np.ndarray,
    sum_cluster: int,
    num_clusters: typing.List[int],
    volume: np.ndarray,
    similarity: np.ndarray,
) -> np.ndarray:
    weight = np.ones(batch.shape[0])
    for i in range(sum_cluster):
        vv = volume * (similarity[i, :])
        ww = np.zeros(num_batch)
        for j in range(num_batch):
            ww[j] = vv[num_clusters[j] : num_clusters[j + 1]].sum()
            if (num_clusters[j] <= i) and (i < num_clusters[j + 1]):
                ww[j] = volume[i]
        ww = ((ww**0.5).sum() ** 2 - ww.sum()) / num_batch / (num_batch - 1)
        ww = ww / volume[i]
        weight[cluster == i] = ww

    weight = weight / weight.mean()

    return weight


def plot_clustering_confidence(
    adata: ad.AnnData, batch_effect: str, ground_truth: str
) -> None:
    mask = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_mask"]
    cluster = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_cluster"]
    sum_cluster = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_sum_cluster"]

    main_cell_types = []
    confidence = np.ndarray(sum_cluster)
    for i in range(sum_cluster):
        curr_label = adata.obs[ground_truth][mask][cluster == i].tolist()
        label_counts = Counter(curr_label)
        most_common = label_counts.most_common(1)
        main_cell_types.append(most_common[0][0])
        confidence[i] = most_common[0][1] / len(curr_label)
    adata.uns[
        config._WEIGHT_PREFIX_ + batch_effect + "_main_cell_types"
    ] = main_cell_types
    adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_confidence"] = confidence

    plt.figure(figsize=(7, 7))
    sns.displot(confidence, kind="ecdf")
    plt.show()


def plot_similarity_confidence(
    adata: ad.AnnData, batch_effect: str, similarity: typing.Callable = _identity
) -> None:
    batch = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_batch"]
    num_batch = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_num_batch"]
    cluster = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_cluster"]
    sum_cluster = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_sum_cluster"]
    num_clusters = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_num_clusters"]
    main_cell_types = adata.uns[
        config._WEIGHT_PREFIX_ + batch_effect + "_main_cell_types"
    ]
    raw_similarity = adata.uns[
        config._WEIGHT_PREFIX_ + batch_effect + "_raw_similarity"
    ]

    truth = raw_similarity.copy()
    mask = raw_similarity.copy()
    for x in range(num_batch):
        for y in range(num_batch):
            for i in range(num_clusters[x], num_clusters[x + 1]):
                for j in range(num_clusters[y], num_clusters[y + 1]):
                    if similarity(main_cell_types[i], main_cell_types[j]):
                        truth[i][j] = 1.0
                    else:
                        truth[i][j] = 0.0
                    if x == y:
                        mask[i][j] = -1.0
                    else:
                        mask[i][j] = 1.0

    print("truth")
    plt.figure(figsize=(7, 7))
    sns.heatmap(data=truth * mask)
    plt.show()

    print("raw_similarity")
    plt.figure(figsize=(7, 7))
    sns.heatmap(data=raw_similarity * mask)
    plt.show()

    coef = 1 - (truth - raw_similarity) ** 2
    print(
        "true positive = %d"
        % ((truth == 1) & (raw_similarity == 1) & (mask == 1)).sum()
    )
    print(
        "true negative = %d"
        % ((truth == 0) & (raw_similarity == 0) & (mask == 1)).sum()
    )
    print(
        "false positive = %d"
        % ((truth == 0) & (raw_similarity == 1) & (mask == 1)).sum()
    )
    print(
        "false negative = %d"
        % ((truth == 1) & (raw_similarity == 0) & (mask == 1)).sum()
    )
    plt.figure(figsize=(7, 7))
    sns.heatmap(data=coef * mask)
    plt.show()


def plot_performance(adata: ad.AnnData, target: str) -> None:
    with rc_context({"figure.figsize": (7, 7)}):
        sc.pl.umap(adata, color=target)


def plot_strategy(
    adata: ad.AnnData,
    batch_effect: str,
    target: str,
    categorical: bool = False,
    log: bool = False,
) -> None:
    adata.obs[target] = adata.uns[config._WEIGHT_PREFIX_ + batch_effect + "_" + target]
    if categorical:
        adata.obs[target] = pd.Series(adata.obs[target], dtype="category")
    if log:
        adata.obs[target] = np.log1p(adata.obs[target])
    with rc_context({"figure.figsize": (7, 7)}):
        sc.pl.umap(adata, color=target)


def get_default_clustering_space(
    adata: ad.AnnData, genes: typing.List[str], batch_effect: str, random_seed: int
) -> np.ndarray:
    adata = data.select_vars(adata, genes)
    x = np.zeros((adata.obs.shape[0], config.PCA_N_COMPONENTS))
    for batch in adata.obs[batch_effect].unique():
        sub_adata = adata[adata.obs[batch_effect] == batch].copy()
        data.normalize(sub_adata)
        sc.pp.log1p(sub_adata)
        sc.pp.pca(sub_adata, n_comps=config.PCA_N_COMPONENTS, random_state=random_seed)
        x[adata.obs[batch_effect] == batch] = sub_adata.obsm["X_pca"]

    return x


def get_default_similarity_space(
    adata: ad.AnnData, genes: typing.List[str], batch_effect: str, random_seed: int
) -> np.ndarray:
    adata = data.select_vars(adata, genes)
    data.normalize(adata)

    return np.log1p(adata.X)


def get_MNN(
    num_batch: int,
    cluster: np.ndarray,
    sum_cluster: int,
    num_clusters: typing.List[int],
    raw_similarity: np.ndarray,
) -> np.ndarray:
    similarity = raw_similarity.copy()
    for i in range(num_batch):
        for j in range(num_batch):
            xy_dist = torch.tensor(
                similarity[
                    num_clusters[i] : num_clusters[i + 1],
                    num_clusters[j] : num_clusters[j + 1],
                ]
            )
            num_x = xy_dist.shape[0]
            num_y = xy_dist.shape[1]
            kx = min(num_x, config.MNN_K)
            ky = min(num_y, config.MNN_K)
            x_topk = F.one_hot(xy_dist.topk(kx, dim=0)[1], num_x).sum(dim=0).bool()
            y_topk = F.one_hot(xy_dist.topk(ky, dim=1)[1], num_y).sum(dim=1).bool()
            mnn_idx = x_topk.T & y_topk
            similarity[
                num_clusters[i] : num_clusters[i + 1],
                num_clusters[j] : num_clusters[j + 1],
            ] = mnn_idx.float().numpy()

    return similarity
