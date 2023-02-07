r"""
Dataset utilities
"""

import os
import typing
import warnings
from collections import OrderedDict

import anndata as ad
import h5py
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import seaborn as sns
import sklearn.metrics
import torch

from . import config, utils


def compute_libsize(adata: ad.AnnData) -> None:
    r"""
    Compute library size

    Parameters
    ----------
    adata
        Input dataset.
    """
    if scipy.sparse.issparse(adata.X):
        adata.obs["__libsize__"] = adata.X.sum(axis=1).A1
    else:
        adata.obs["__libsize__"] = adata.X.sum(axis=1)


def normalize(adata: ad.AnnData, target: float = 10000.0) -> None:
    r"""
    Obs-wise normalization of expression matrix.

    Parameters
    ----------
    adata
        Input dataset.
    target
        Target value of normalization.
    """
    if "__libsize__" not in adata.obs.columns:
        compute_libsize(adata)
    normalizer = target / np.expand_dims(adata.obs["__libsize__"].to_numpy(), axis=1)
    adata.X = (
        adata.X.multiply(normalizer).tocsr()
        if scipy.sparse.issparse(adata.X)
        else adata.X * normalizer
    )


def find_variable_genes(
    adata: ad.AnnData,
    slot: str = "variable_genes",
    x_low_cutoff: float = 0.1,
    x_high_cutoff: float = 8.0,
    y_low_cutoff: float = 1.0,
    y_high_cutoff: float = np.inf,
    num_bin: int = 20,
    binning_method: str = "equal_frequency",
    grouping: typing.Optional[str] = None,
    min_group_frac: float = 0.5,
) -> typing.Union[matplotlib.axes.Axes, typing.Mapping[str, matplotlib.axes.Axes]]:
    r"""
    A reimplementation of the Seurat v2 "mean.var.plot" gene selection
    method in the "FindVariableGenes" function, with the extended ability
    of selecting variable genes within specified groups of cells and then
    combine results of individual groups. This is useful to minimize batch
    effect during feature selection.

    Parameters
    ----------
    adata
        Input dataset
    slot
        Slot in `var` to store the variable genes
    x_low_cutoff
        Minimal log mean cutoff
    x_high_cutoff
        Maximal log mean cutoff
    y_low_cutoff
        Minimal log VMR cutoff
    y_high_cutoff
        Maximal log VMR cutoff
    num_bin
        Number of bins based on mean expression.
    binning_method
        How binning should be done based on mean expression.
        Available choices include {"equal_width", "equal_frequency"}.
    grouping
        Specify a column in the ``obs`` table that splits cells into
        several groups. Gene selection is performed in each group separately
        and results are combined afterwards.
    min_group_frac
        The minimal fraction of groups in which a gene must be selected
        for it to be kept in the final result.
    Returns
    -------
    ax
        VMR plot (a dict of plots if grouping is specified)
    """
    if grouping is not None:
        ax_dict = {}
        selected_list = []
        groups = np.unique(adata.obs[grouping])
        for group in groups:
            tmp_adata = adata[adata.obs[grouping] == group, :].copy()
            ax_dict[group] = find_variable_genes(
                tmp_adata,
                slot=slot,
                x_low_cutoff=x_low_cutoff,
                x_high_cutoff=x_high_cutoff,
                y_low_cutoff=y_low_cutoff,
                y_high_cutoff=y_high_cutoff,
                num_bin=num_bin,
                binning_method=binning_method,
            )
            selected_list.append(tmp_adata.var[slot].to_numpy().ravel())
        selected_count = np.stack(selected_list, axis=1).sum(axis=1)
        adata.var[slot] = False
        adata.var[slot].loc[selected_count >= min_group_frac * groups.size] = True
        return ax_dict

    X_backup = adata.X
    normalize(adata)
    X = adata.X
    mean = np.asarray(np.mean(X, axis=0)).ravel()
    var = np.asarray(
        np.mean(X.power(2) if scipy.sparse.issparse(X) else np.square(X), axis=0)
    ).ravel() - np.square(mean)
    log_mean = np.log1p(mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_vmr = np.log(var / mean)
    log_vmr[np.isnan(log_vmr)] = 0
    if binning_method == "equal_width":
        log_mean_bin = pd.cut(log_mean, num_bin)
    elif binning_method == "equal_frequency":
        log_mean_bin = pd.cut(
            log_mean,
            [-1]
            + np.percentile(
                log_mean[log_mean > 0], np.linspace(0, 100, num_bin)
            ).tolist(),
        )
    else:
        raise ValueError("Invalid binning method!")
    summary_df = pd.DataFrame(
        {"log_mean": log_mean, "log_vmr": log_vmr, "log_mean_bin": log_mean_bin},
        index=adata.var_names,
    )
    summary_df["log_vmr_scaled"] = (
        summary_df.loc[:, ["log_vmr", "log_mean_bin"]]
        .groupby("log_mean_bin")
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    summary_df["log_vmr_scaled"].fillna(0, inplace=True)
    selected = summary_df.query(
        f"log_mean > {x_low_cutoff} & log_mean < {x_high_cutoff} & "
        f"log_vmr_scaled > {y_low_cutoff} & log_vmr_scaled < {y_high_cutoff}"
    )
    summary_df["selected"] = np.in1d(summary_df.index, selected.index)
    adata.var[slot] = False
    adata.var[slot].loc[selected.index] = True
    adata.X = X_backup

    _, ax = plt.subplots(figsize=(7, 7))
    ax = sns.scatterplot(
        x="log_mean",
        y="log_vmr_scaled",
        hue="selected",
        data=summary_df,
        edgecolor=None,
        s=5,
        ax=ax,
    )
    for _, row in selected.iterrows():
        ax.text(
            row["log_mean"],
            row["log_vmr_scaled"],
            row.name,
            size="x-small",
            ha="center",
            va="center",
        )
    ax.set_xlabel("Average expression")
    ax.set_ylabel("Dispersion")
    return ax


def _expanded_subset(
    mat: typing.Union[scipy.sparse.spmatrix, np.ndarray],
    idx: np.ndarray,
    axis: int = 0,
    fill: typing.Any = 0,
) -> typing.Union[scipy.sparse.spmatrix, np.ndarray]:
    assert axis in (0, 1)
    expand_size = max(idx.max() - mat.shape[axis] + 1, 0)
    if axis == 0:
        if scipy.sparse.issparse(mat):
            expand_mat = scipy.sparse.lil_matrix(
                (expand_size, mat.shape[1]), dtype=mat.dtype
            )
            if fill != 0:
                expand_mat[:] = fill
            expand_mat = scipy.sparse.vstack([mat.tocsr(), expand_mat.tocsr()])
        else:
            expand_mat = np.empty((expand_size, mat.shape[1]), dtype=mat.dtype)
            expand_mat[:] = fill
            expand_mat = np.concatenate([mat, expand_mat], axis=0)
        result_mat = expand_mat[idx, :]
    else:
        if scipy.sparse.issparse(mat):
            expand_mat = scipy.sparse.lil_matrix(
                (mat.shape[0], expand_size), dtype=mat.dtype
            )
            if fill != 0:
                expand_mat[:] = fill
            expand_mat = scipy.sparse.hstack([mat.tocsc(), expand_mat.tocsc()])
        else:
            expand_mat = np.empty((mat.shape[0], expand_size), dtype=mat.dtype)
            expand_mat[:] = fill
            expand_mat = np.concatenate([mat, expand_mat], axis=1)
        result_mat = expand_mat[:, idx]
    if scipy.sparse.issparse(result_mat):
        result_mat = result_mat.tocsr()
    return result_mat


def select_vars(adata: ad.AnnData, var_names: typing.List[str]) -> ad.AnnData:
    r"""
    Select variables with special support for variables inexistent in the input
    (in which case the inexistent variables will be filled with zeros).

    Note that "raw", "varm" and "layers" will be discarded.

    Parameters
    ----------
    adata
        Input dataset.
    var_names
        Variables to select.

    Returns
    -------
    selected
        Dataset with selected variables.
    """
    if adata.var_names.duplicated().any():
        raise ValueError("Variable names are not unique!")
    new_var_names = np.setdiff1d(np.unique(var_names), adata.var_names)
    all_var_names = np.concatenate([adata.var_names.to_numpy(), new_var_names])
    if new_var_names.size > 0:  # pragma: no cover
        utils.logger.warning(
            "%d out of %d variables are not found, will be set to zero!",
            len(new_var_names),
            len(var_names),
        )
        utils.logger.info(str(new_var_names.tolist()).strip("[]"))
    idx = np.vectorize(lambda x: np.where(all_var_names == x)[0][0])(var_names)

    new_X = _expanded_subset(adata.X, idx, axis=1, fill=0)
    new_var = adata.var.reindex(var_names)
    return ad.AnnData(
        X=new_X, obs=adata.obs, var=new_var, uns=adata.uns, obsm=adata.obsm
    )


def map_vars(
    adata: ad.AnnData,
    mapping: pd.DataFrame,
    map_hvg: typing.Optional[typing.List[str]] = None,
) -> ad.AnnData:
    r"""
    Map variables of input dataset to some other terms,
    e.g. gene ortholog groups, or orthologous genes in another species.

    Note that "raw", "varm" and "layers" will be discarded.

    Parameters
    ----------
    adata
        Input dataset.
    mapping
        A 2-column data frame defining variable name mapping. First column
        is source variable name and second column is target variable name.
    map_hvg
        Specify `var` slots containing highly variable genes
        that should also be mapped.

    Returns
    -------
    mapped
        Mapped dataset.
    """
    # Convert to mapping matrix
    if adata.var_names.duplicated().any():
        raise ValueError("Variable names are not unique!")
    source = adata.var_names
    mapping = mapping.loc[np.in1d(mapping.iloc[:, 0], source), :]
    target = np.unique(mapping.iloc[:, 1])

    source_idx_map = {val: i for i, val in enumerate(source)}
    target_idx_map = {val: i for i, val in enumerate(target)}
    source_idx = [source_idx_map[val] for val in mapping.iloc[:, 0]]
    target_idx = [target_idx_map[val] for val in mapping.iloc[:, 1]]
    mapping = scipy.sparse.csc_matrix(
        (np.repeat(1, mapping.shape[0]), (source_idx, target_idx)),
        shape=(source.size, target.size),
    )

    # Sanity check
    amb_src_mask = np.asarray(mapping.sum(axis=1)).squeeze() > 1
    amb_tgt_mask = np.asarray(mapping.sum(axis=0)).squeeze() > 1
    if amb_src_mask.sum() > 0:  # pragma: no cover
        utils.logger.warning("%d ambiguous source items found!", amb_src_mask.sum())
        utils.logger.info(str(source[amb_src_mask].tolist()))
    if amb_tgt_mask.sum() > 0:  # pragma: no cover
        utils.logger.warning("%d ambiguous target items found!", amb_tgt_mask.sum())
        utils.logger.info(str(target[amb_tgt_mask].tolist()))

    # Compute new expression matrix
    new_X = adata.X @ mapping
    if scipy.sparse.issparse(new_X):
        new_X = new_X.tocsr()

    # Update var accordingly
    if map_hvg:
        new_var = adata.var.loc[:, map_hvg].to_numpy().astype(int).T
        new_var = (new_var @ mapping).T.astype(bool)
        new_var = pd.DataFrame(new_var, columns=map_hvg, index=target)
    else:
        new_var = pd.DataFrame(index=target)

    return ad.AnnData(
        X=new_X, obs=adata.obs, var=new_var, uns=adata.uns, obsm=adata.obsm
    )


def annotation_confidence(
    adata: ad.AnnData,
    annotation: typing.Union[str, typing.List[str]],
    used_vars: typing.Optional[typing.List[str]] = None,
    metric: str = "cosine",
    return_group_percentile: bool = True,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute annotation confidence of each obs (cell) based on
    sample silhouette score.

    Parameters
    ----------
    adata
        Input dataset
    annotation
        Specifies annotation for which confidence will be computed.
        If passed an array-like, it should be 1 dimensional with length
        equal to obs number, and will be used directly as annotation.
        If passed a string, it should be a column name in ``obs``.
    used_vars
        Specifies the variables used to evaluate ``metric``,
        If not specified, all variables are used.
    metric
        Specifies distance metric used to compute sample silhouette scores.
        See :func:`sklearn.metrics.silhouette_samples` for available
        options.
    return_group_percentile
        Whether to return within group confidence percentile, instead of
        raw sample silhouette score.

    Returns
    -------
    confidence
        1 dimensional numpy array containing annotation confidence for
        each obs.
    group_percentile
        1 dimensional numpy array containing within-group percentile
        for each obs.
    """
    if isinstance(annotation, str):
        annotation = adata.obs[annotation].to_numpy()
    annotation = utils.encode_integer(annotation)[0]
    if used_vars is None:
        used_vars = adata.var_names
    X = select_vars(adata, used_vars).X
    X = np.log1p(X)
    confidence = sklearn.metrics.silhouette_samples(X, annotation, metric=metric)
    if return_group_percentile:
        normalized_confidence = np.zeros_like(confidence)
        for l in np.unique(annotation):
            mask = annotation == l
            normalized_confidence[mask] = (
                scipy.stats.rankdata(confidence[mask]) - 1
            ) / (mask.sum() - 1)
        return confidence, normalized_confidence
    return confidence


def write_table(
    adata: ad.AnnData, filename: str, orientation: str = "cg", **kwargs
) -> None:
    r"""
    Write the expression matrix to a plain-text file.
    Note that ``obs`` (cell) meta table, ``var`` (gene) meta table and data
    in the ``uns`` slot are discarded, only the expression matrix is written
    to the file.

    Parameters
    ----------
    adata
        Input Dataset.
    filename
        Name of the file to be written.
    orientation
        Specifies whether to write in :math:`obs \times var` or
        :math:`obs \times var` orientation, should be among {"cg", "gc"}.
    kwargs
        Additional keyword arguments will be passed to
        :meth:`pandas.DataFrame.to_csv`.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if orientation == "cg":
        df = pd.DataFrame(
            utils.densify(adata.X), index=adata.obs_names, columns=adata.var_names
        )
    elif orientation == "gc":
        df = pd.DataFrame(
            utils.densify(adata.X.T), index=adata.var_names, columns=adata.obs_names
        )
    else:  # pragma: no cover
        raise ValueError("Invalid orientation!")
    df.to_csv(filename, **kwargs)


def read_table(
    filename: str, orientation: str = "cg", sparsify: bool = False, **kwargs
) -> ad.AnnData:
    r"""
    Read expression matrix from a plain-text file

    Parameters
    ----------
    filename
        Name of the file to read from.
    orientation
        Specifies whether matrix in the file is in
        :math:`cell \times gene` or :math:`gene \times cell` orientation.
    sparsify
        Whether to convert the expression matrix into sparse format.
    kwargs
        Additional keyword arguments will be passed to :func:`pandas.read_csv`.
    Returns
    -------
    loaded_dataset
        An :class:`ad.AnnData` object loaded from the file.
    """
    df = pd.read_csv(filename, **kwargs)
    if orientation == "gc":
        df = df.T
    return ad.AnnData(
        scipy.sparse.csr_matrix(df.values) if sparsify else df.values,
        pd.DataFrame(index=df.index),
        pd.DataFrame(index=df.columns),
        {},
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: typing.OrderedDict) -> None:
        super().__init__()
        self.data_dict = data_dict
        for key, value in self.data_dict.items():
            self.data_dict[key] = torch.tensor(utils.densify(value)).float()

    def __getitem__(self, idx):
        if isinstance(idx, (slice, np.ndarray)):
            return OrderedDict(
                [
                    # (item, torch.tensor(utils.densify(self.data_dict[item][idx])).float()) for item in self.data_dict
                    (item, self.data_dict[item][idx])
                    for item in self.data_dict
                ]
            )
        elif isinstance(idx, int):
            return OrderedDict(
                [
                    # (item, torch.tensor(utils.densify(self.data_dict[item][idx])).squeeze().float()) for item in self.data_dict
                    (item, self.data_dict[item][idx])
                    for item in self.data_dict
                ]
            )
        return self.data_dict[idx]

    def __len__(self):
        data_size = set([item.shape[0] for item in self.data_dict.values()])
        if data_size:
            assert len(data_size) == 1
            return data_size.pop()
        return 0


def h5_to_h5ad(inputfile: str, outputfile: str):
    with h5py.File(inputfile, "r") as f:
        obs = pd.DataFrame(
            dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...])
        )
        var = pd.DataFrame(
            dict_from_group(f["var"]), index=utils.decode(f["var_names"][...])
        )
        uns = dict_from_group(f["uns"])

        exprs_handle = f["exprs"]
        if isinstance(exprs_handle, h5py.Group):  # Sparse matrix
            mat = scipy.sparse.csr_matrix(
                (
                    exprs_handle["data"][...],
                    exprs_handle["indices"][...],
                    exprs_handle["indptr"][...],
                ),
                shape=exprs_handle["shape"][...],
            )
        else:  # Dense matrix
            mat = exprs_handle[...].astype(np.float32)

    adata = ad.AnnData(X=mat, obs=obs, var=var, uns=dict(uns))
    adata.write(outputfile)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
        mask = data == config._NAN_REPLACEMENT
        if np.any(mask):
            data = data.astype(object)
            data[mask] = np.nan
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d
