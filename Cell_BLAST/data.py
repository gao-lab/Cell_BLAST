r"""
Dataset utilities
"""

import collections
import concurrent.futures
import copy
import functools
import os
import typing

import anndata
import h5py
import loompy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import seaborn as sns
import sklearn.metrics

from . import config, utils


class ExprDataSet(object):

    r"""
    Main data class, which is based on the data structure of
    :class:`anndata.AnnData`.
    Note that the data is always assumed to be scRNA-seq, so the stored
    matrix is always the expression matrix. The ``obs`` slot stores meta
    information of cells, and the ``var`` slot stores meta information of genes.
    The ``uns`` slot stores other unstructure data, e.g. list of most
    informative genes, etc.

    Parameters
    ----------
    exprs
        An :math:`obs \times var` expression matrix in the form of
        either a numpy array or a scipy sparse matrix.
    obs
        Cell meta table. Each row corresponds to a row in ``exprs``.
    var
        Gene meta table. Each row corresponds to a column in ``exprs``.
    uns
        Unstructured meta information, e.g. list of most informative genes.

    Examples
    --------

    An :class:`ExprDataSet` object can be constructed from an expression matrix,
    an observation (cell) meta table, a variable (gene) meta table, and some
    unstructured data:

    >>> data_obj = Cell_BLAST.data.ExprDataSet(exprs, obs, var, uns)

    Or, if you have an :class:`anndata.AnnData` object or a
    :class:`loompy.loompy.LoomConnection` object (to a loom file), you can
    directly convert them to an :class:`ExprDataSet` object using the following
    methods:

    >>> data_obj = Cell_BLAST.data.ExprDataSet.from_anndata(anndata_obj)
    >>> data_obj = Cell_BLAST.data.ExprDataSet.from_loom(loomconnection_obj)

    It's also possible to convert in the opposite direction:

    >>> anndata_obj = data_obj.to_anndata()
    >>> loomconnection = data_obj.to_loom(filename)

    :class:`ExprDataSet` objects support many forms of slicing, including
    numeric range, numeric index, boolean mask, and obs/var name selection:

    >>> subdata_obj = data_obj[0:10, np.arange(10)]
    >>> subdata_obj = data_obj[
    ...     data_obj.obs["cell_ontology_class"] == "endothelial",
    ...     ["gene_1", "gene_2"]
    ... ]

    Note that in variable name selection, if a variable does not exist in the
    original dataset, it will be filled with zeros in the returned dataset,
    with a warning message.

    :class:`ExprDataSet` objects also support saving and loading:

    >>> data_obj.write_dataset("data.h5")
    >>> data_obj = Cell_BLAST.data.ExprDataSet.read_dataset("data.h5")

    Some other utilities used in the Cell_BLAST pipeline are also supported,
    including but not limited to:

    Dataset merging

    >>> combined_data_obj = Cell_BLAST.data.ExprDataSet.merge_datasets({
    ...     "data1": data_obj1,
    ...     "data2": data_obj2,
    ...     "data3": data_obj3
    ... })

    Data visualization

    >>> data_obj.latent = latent_matrix
    >>> ax = data_obj.visualize_latent("cell_type")
    >>> ax = data_obj.violin("cell_type", "gene_name")
    >>> ax = data_obj.obs_correlation_heatmap()

    Find markers:

    >>> marker_dict = data_obj.fast_markers("cell_type")
    """

    def __init__(
            self, exprs: typing.Union[np.ndarray, scipy.sparse.spmatrix],
            obs: pd.DataFrame, var: pd.DataFrame, uns: typing.Mapping
    ) -> None:
        assert exprs.shape[0] == obs.shape[0] and exprs.shape[1] == var.shape[0]
        if scipy.sparse.issparse(exprs):
            self.exprs = exprs.tocsr()
        else:
            self.exprs = exprs
        self.obs = obs
        self.var = var
        self.uns = uns

    @property
    def X(self) -> typing.Union[np.ndarray, scipy.sparse.spmatrix]:  # For compatibility with `AnnData`
        r"""
        :math:`obs \times var` expression matrix, same as ``exprs``
        """
        return self.exprs

    @property
    def obs_names(self) -> pd.Index:
        r"""
        Name of observations (cells)
        """
        return self.obs.index

    @obs_names.setter
    def obs_names(self, new_names: np.ndarray) -> None:
        assert len(new_names) == self.obs.shape[0]
        self.obs.index = new_names

    @property
    def var_names(self) -> pd.Index:
        r"""
        Name of variables (genes)
        """
        return self.var.index

    @var_names.setter
    def var_names(self, new_names: np.ndarray) -> None:
        assert len(new_names) == self.var.shape[0]
        self.var.index = new_names

    @property
    def shape(self) -> typing.Tuple[int, int]:
        r"""
        Shape of dataset (:math:`obs \times var`)
        """
        return self.exprs.shape

    @property
    def latent(self) -> np.ndarray:
        r"""
        Latent space coordinate. Must have the same number of observations
        (cells) as the expression data.
        """
        mask = np.vectorize(lambda x: x.startswith("latent_"))(self.obs.columns)
        if np.any(mask):
            return self.obs.loc[:, np.vectorize(lambda x: f"latent_{x}")(
                np.arange(mask.sum()) + 1
            )].values
        else:
            raise ValueError("No latent has been registered!")

    @latent.setter
    def latent(self, latent: np.ndarray) -> None:
        for col in self.obs.columns:  # Remove previous result
            if col.startswith("latent_"):
                del self.obs[col]
            if col.startswith("tSNE"):
                del self.obs[col]
            if col.startswith("UMAP"):
                del self.obs[col]
        assert latent.shape[0] == self.shape[0]
        columns = np.vectorize(
            lambda x: f"latent_{x}"
        )(np.arange(latent.shape[1]) + 1)
        latent_df = pd.DataFrame(latent, index=self.obs_names, columns=columns)
        self.obs = pd.concat([self.obs, latent_df], axis=1)

    def normalize(self, target: float = 10000.0) -> "ExprDataSet":
        r"""
        Obs-wise (cell-wise) normalization if the expression matrix.
        Note that only the matrix gets copied in the returned dataset, but meta
        tables are not (only references to meta tables in the original dataset).

        Parameters
        ----------
        target
            Target value of normalization.

        Returns
        -------
        normalized
            Normalized ExprDataSet object.
        """
        import sklearn.preprocessing
        tmp = self.copy()
        if "__libsize__" not in self.obs.columns:
            tmp.exprs = sklearn.preprocessing.normalize(
                tmp.exprs, norm="l1", copy=True
            ) * target
        else:
            normalizer = target / np.expand_dims(
                tmp.obs["__libsize__"].to_numpy(), axis=1)
            tmp.exprs = tmp.exprs.multiply(normalizer) \
                if scipy.sparse.issparse(tmp.exprs) \
                else tmp.exprs * normalizer
        return tmp

    def __getitem__(self, slices) -> "ExprDataSet":
        r"""
        2-d slicing by numeric range, numeric index, boolean mask and obs/var names.
        """
        obs_slice, var_slice = slices

        # var splicing
        simple_flag = False
        if isinstance(var_slice, slice):
            simple_flag = True
        else:  # iterable
            var_slice = np.array(var_slice)
            if var_slice.ndim == 0:
                var_slice = np.expand_dims(var_slice, axis=0)
            if var_slice.size == 0:
                var_slice = var_slice.astype(int)
                simple_flag = True
            if np.issubdtype(var_slice.dtype.type, np.bool_):
                var_slice = np.where(var_slice)[0]
            if np.issubdtype(var_slice.dtype.type, np.integer):
                simple_flag = True

        if simple_flag:
            exprs = self.exprs.tocsc()[:, var_slice].tocsr() \
                if scipy.sparse.issparse(self.exprs) else self.exprs[:, var_slice]
            var = self.var.iloc[var_slice, :]
        else:  # Name slicing
            new_var_names = np.setdiff1d(var_slice, self.var_names)
            all_var_names = np.concatenate([
                self.var_names.values, new_var_names])
            if new_var_names.size > 0:  # pragma: no cover
                utils.logger.warning(
                    "%d out of %d variables are not found, will be set to zero!",
                    len(new_var_names), len(var_slice)
                )
                utils.logger.info(str(new_var_names.tolist()).strip("[]"))
            idx = np.vectorize(
                lambda x: np.where(all_var_names == x)[0][0]
            )(var_slice)
            exprs = scipy.sparse.hstack([
                self.exprs.tocsc(),
                scipy.sparse.csc_matrix((
                    self.exprs.shape[0], len(new_var_names)
                ), dtype=self.exprs.dtype)
            ])[:, idx].tocsr() if scipy.sparse.issparse(self.exprs) else np.concatenate([
                self.exprs, np.zeros((
                    self.exprs.shape[0], len(new_var_names)
                ), dtype=self.exprs.dtype)
            ], axis=1)[:, idx]
            var = self.var.reindex(var_slice)

        # obs slicing
        simple_flag = False
        if isinstance(obs_slice, slice):
            simple_flag = True
        else:  # iterable
            obs_slice = np.array(obs_slice)
            if obs_slice.ndim == 0:
                obs_slice = np.expand_dims(obs_slice, axis=0)
            if obs_slice.size == 0:
                obs_slice = obs_slice.astype(int)
                simple_flag = True
            if issubclass(obs_slice.dtype.type, np.bool_):
                obs_slice = np.where(obs_slice)[0]
            if issubclass(obs_slice.dtype.type, np.integer):
                simple_flag = True

        if simple_flag:
            exprs = exprs[obs_slice, :]
            obs = self.obs.iloc[obs_slice, :]
        else:  # Name slicing
            assert np.all(np.in1d(obs_slice, self.obs_names))
            extract_idx = np.vectorize(
                lambda x: np.where(self.obs_names == x)[0][0]
            )(obs_slice)
            exprs = exprs[extract_idx, :]
            obs = self.obs.reindex(obs_slice)

        return ExprDataSet(exprs=exprs, obs=obs, var=var, uns=self.uns)

    def clean_duplicate_vars(self) -> "ExprDataSet":
        r"""
        Clean up variables to preserve only the first occurrence of
        duplicated variables.

        Returns
        -------
        cleaned
            An ExprDataSet object with duplicated variables removed.
        """
        unique_vars, duplicate_mask = \
            set(), np.ones(self.var_names.size).astype(np.bool_)
        for idx, item in enumerate(self.var_names):
            if item in unique_vars:
                duplicate_mask[idx] = False
            else:
                unique_vars.add(item)
        return self[:, duplicate_mask]

    def find_variable_genes(
            self,
            x_low_cutoff: float = 0.1,
            x_high_cutoff: float = 8.0,
            y_low_cutoff: float = 1.0,
            y_high_cutoff: float = np.inf,
            num_bin: int = 20,
            binning_method: str = "equal_frequency",
            grouping: typing.Optional[str] = None,
            min_group_frac: float = 0.5,
    ) -> typing.Tuple[
        typing.List[str],
        typing.Union[matplotlib.axes.Axes, typing.Mapping[str, matplotlib.axes.Axes]]
    ]:
        r"""
        A reimplementation of the Seurat v2 "mean.var.plot" gene selection
        method in the "FindVariableGenes" function, with the extended ability
        of selecting variable genes within specified groups of cells and then
        combine results of individual groups. This is useful to minimize batch
        effect during feature selection.

        Parameters
        ----------
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
        list
            A list of selected variables
        ax
            VMR plot (a dict of plots if grouping is specified)
        """
        if grouping is not None:
            selected_dict, ax_dict = {}, {}
            groups = np.unique(self.obs[grouping])
            for group in groups:
                result = self[
                    self.obs[grouping] == group, :
                ].find_variable_genes(
                    x_low_cutoff=x_low_cutoff, x_high_cutoff=x_high_cutoff,
                    y_low_cutoff=y_low_cutoff, y_high_cutoff=y_high_cutoff,
                    num_bin=num_bin, binning_method=binning_method
                )
                selected_dict[group] = result[0]
                ax_dict[group] = result[1]
            selected = np.concatenate(list(selected_dict.values()))
            selected_unique, selected_count = np.unique(selected, return_counts=True)
            selected = selected_unique[selected_count >= min_group_frac * groups.size]
            return selected, ax_dict

        exprs = self.normalize().exprs
        mean = np.asarray(np.mean(exprs, axis=0)).ravel()
        var = np.asarray(np.mean(
            exprs.power(2) if scipy.sparse.issparse(exprs)
            else np.square(exprs),
            axis=0
        )).ravel() - np.square(mean)
        log_mean = np.log1p(mean)
        log_vmr = np.log(var / mean)
        log_vmr[np.isnan(log_vmr)] = 0
        if binning_method == "equal_width":
            log_mean_bin = pd.cut(log_mean, num_bin)
        elif binning_method == "equal_frequency":
            log_mean_bin = pd.cut(
                log_mean, [-1] + np.percentile(
                    log_mean[log_mean > 0], np.linspace(0, 100, num_bin)
                ).tolist()
            )
        else:
            raise ValueError("Invalid binning method!")
        summary_df = pd.DataFrame({
            "log_mean": log_mean,
            "log_vmr": log_vmr,
            "log_mean_bin": log_mean_bin
        }, index=self.var_names)
        summary_df["log_vmr_scaled"] = summary_df.loc[
            :, ["log_vmr", "log_mean_bin"]
        ].groupby("log_mean_bin").transform(lambda x: (x - x.mean()) / x.std())
        summary_df["log_vmr_scaled"].fillna(0, inplace=True)
        selected = summary_df.query(
            f"log_mean > {x_low_cutoff} & log_mean < {x_high_cutoff} & "
            f"log_vmr_scaled > {y_low_cutoff} & log_vmr_scaled < {y_high_cutoff}"
        )
        summary_df["selected"] = np.in1d(summary_df.index, selected.index)

        _, ax = plt.subplots(figsize=(7, 7))
        ax = sns.scatterplot(
            x="log_mean", y="log_vmr_scaled", hue="selected",
            data=summary_df, edgecolor=None, s=5, ax=ax
        )
        for _, row in selected.iterrows():
            ax.text(
                row["log_mean"], row["log_vmr_scaled"], row.name,
                size="x-small", ha="center", va="center"
            )
        ax.set_xlabel("Average expression")
        ax.set_ylabel("Dispersion")
        return selected.index.to_numpy().tolist(), ax

    def get_meta_or_var(
            self, names: typing.List[str], normalize_var: bool = False,
            log_var: bool = False
    ) -> pd.DataFrame:
        r"""
        Get either cell meta information (specified by column names in
        the ``obs`` table) or gene expression values in the expression matrix
        (specified by gene names).

        Parameters
        ----------
        names
            List of names that specifies meta information / genes to be fetched.
        normalize_var
            Whether to do cell-normalization before fetching variable values.
        log_var
            Whether to apply log transform for fetched variable values.

        Returns
        -------
        fetched
            Fetched result.
        """
        meta_names = np.intersect1d(names, self.obs.columns)
        remain_names = np.setdiff1d(names, self.obs.columns)
        var_names = np.intersect1d(remain_names, self.var_names)
        if var_names.size != remain_names.size:  # pragma: no cover
            raise ValueError("Unknown names encountered!")
        result = self.obs.loc[:, meta_names]
        if var_names.size:
            ds = self.normalize() if normalize_var else self
            exprs = utils.densify(ds[:, var_names].exprs)
            exprs = np.log1p(exprs) if log_var else exprs
            var = pd.DataFrame(exprs, index=self.obs_names, columns=var_names)
            result = pd.concat([result, var], axis=1)
        return result.loc[:, names]

    def copy(self, deep: bool = False) -> "ExprDataSet":
        r"""
        Produce a copy of the dataset.

        Parameters
        ----------
        deep
            Whether to perform deep copy.

        Returns
        -------
        copied
            Copy of the dataset.
        """
        if deep:
            return ExprDataSet(
                self.exprs.copy(), self.obs.copy(), self.var.copy(),
                copy.deepcopy(self.uns)
            )
        return ExprDataSet(self.exprs, self.obs, self.var, self.uns)

    def write_dataset(self, filename: str) -> None:
        r"""
        Write the dataset to a file.

        Parameters
        ----------
        filename
            File to be written (content in hdf5 format).
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        h5opts = {**config.H5_COMPRESS_OPTS, **config.H5_TRACK_OPTS}
        with h5py.File(filename, "w") as f:
            if scipy.sparse.issparse(self.exprs):
                self.exprs.sort_indices()  # Compatibility with R
                g = f.create_group("exprs")
                g.create_dataset("data", data=self.exprs.data, **h5opts)
                g.create_dataset("indices", data=self.exprs.indices, **h5opts)
                g.create_dataset("indptr", data=self.exprs.indptr, **h5opts)
                g.create_dataset("shape", data=self.exprs.shape, **h5opts)
            else:
                f.create_dataset("exprs", data=self.exprs, **h5opts)
            f.create_dataset("obs_names", data=utils.encode(self.obs_names.values), **h5opts)
            f.create_dataset("var_names", data=utils.encode(self.var_names.values), **h5opts)
            dict_to_group(df_to_dict(self.obs), f.create_group("obs"))
            dict_to_group(df_to_dict(self.var), f.create_group("var"))
            dict_to_group(self.uns, f.create_group("uns"))

    @classmethod
    def read_dataset(
            cls, filename: str, sparsify: bool = False,
            skip_exprs: bool = False
    ) -> "ExprDataSet":
        r"""
        Read dataset from file (saved by ``write_dataset``).

        Parameters
        ----------
        filename
            File to read from (content in hdf5 format).
        sparsify
            Whether to coerce the expression matrix into sparse format.
        skip_exprs
            Whether to skip reading the expression matrix (fill with all
            zeros instead). This option is for accelerating data reading
            in case if only the meta information are needed.

        Returns
        -------
        loaded_dataset
            An ExprDataSet object loaded from the file.
        """
        with h5py.File(filename, "r") as f:
            obs = pd.DataFrame(
                dict_from_group(f["obs"]),
                index=utils.decode(f["obs_names"][...])
            )
            var = pd.DataFrame(
                dict_from_group(f["var"]),
                index=utils.decode(f["var_names"][...])
            )
            uns = dict_from_group(f["uns"])

            if not skip_exprs:
                exprs_handle = f["exprs"]
                if isinstance(exprs_handle, h5py.Group):  # Sparse matrix
                    mat = scipy.sparse.csr_matrix((
                        exprs_handle['data'][...],
                        exprs_handle['indices'][...],
                        exprs_handle['indptr'][...]
                    ), shape=exprs_handle['shape'][...])
                else:  # Dense matrix
                    mat = exprs_handle[...].astype(np.float32)
                    if sparsify:
                        mat = scipy.sparse.csr_matrix(mat)
            else:
                mat = scipy.sparse.csr_matrix((obs.shape[0], var.shape[0]))
        return cls(exprs=mat, obs=obs, var=var, uns=uns)

    def map_vars(
            self, mapping: pd.DataFrame,
            map_uns_slots: typing.Optional[typing.List[str]] = None
    ) -> "ExprDataSet":
        r"""
        Map variables of the dataset to some other terms,
        e.g. gene ortholog groups, or orthologous genes in another species.

        Parameters
        ----------
        mapping
            A 2-column data frame defining variable name mapping. First column
            is source variable name and second column is target variable name.
        map_uns_slots
            Assuming variable subsets, e.g. most informative genes,
            are stored in the ``uns`` slot, this parameter specifies which slots
            in ``uns`` should also be mapped.
            Note that ``uns`` slots not specified here will be left as is.

        Returns
        -------
        mapped
            Mapped dataset.
        """
        # Convert to mapping matrix
        source = self.var_names
        mapping = mapping.loc[np.in1d(mapping.iloc[:, 0], source), :]
        target = np.unique(mapping.iloc[:, 1])

        source_idx_map = {val: i for i, val in enumerate(source)}
        target_idx_map = {val: i for i, val in enumerate(target)}
        source_idx = [source_idx_map[val] for val in mapping.iloc[:, 0]]
        target_idx = [target_idx_map[val] for val in mapping.iloc[:, 1]]
        mapping = scipy.sparse.csc_matrix(
            (np.repeat(1, mapping.shape[0]), (source_idx, target_idx)),
            shape=(source.size, target.size)
        )

        # Sanity check
        amb_src_mask = np.asarray(mapping.sum(axis=1)).squeeze() > 1
        amb_tgt_mask = np.asarray(mapping.sum(axis=0)).squeeze() > 1
        if amb_src_mask.sum() > 0:  # pragma: no cover
            utils.logger.warning("%d ambiguous source items found!",
                                 amb_src_mask.sum())
            utils.logger.info(str(source[amb_src_mask].tolist()))
        if amb_tgt_mask.sum() > 0:  # pragma: no cover
            utils.logger.warning("%d ambiguous target items found!",
                                 amb_tgt_mask.sum())
            utils.logger.info(str(target[amb_tgt_mask].tolist()))

        # Compute new expression matrix
        new_exprs = self.exprs @ mapping
        if scipy.sparse.issparse(self.exprs):
            new_exprs = scipy.sparse.csr_matrix(new_exprs)

        # Update var accordingly
        new_var = pd.DataFrame(index=target)

        # Update uns accordingly
        if map_uns_slots is None:
            map_uns_slots = []
        new_uns = {}
        for slot in self.uns:
            if slot in map_uns_slots:
                idx = [source_idx_map[val] for val in self.uns[slot]]
                idx = np.where(np.asarray(
                    mapping[idx, :].sum(axis=0)
                ).squeeze() > 0)[0]
                new_uns[slot] = target[idx]
            else:
                new_uns[slot] = self.uns[slot]
        return ExprDataSet(new_exprs, self.obs, new_var, new_uns)

    @classmethod
    def merge_datasets(
            cls, dataset_dict: typing.Mapping[str, "ExprDataSet"],
            meta_col: typing.Optional[str] = None,
            merge_uns_slots: typing.Optional[typing.List[str]] = None
    ) -> "ExprDataSet":
        r"""
        Merge multiple dataset objects into a single "meta-dataset".

        Parameters
        ----------
        dataset_dict
            A dict of ExprDataSet objects. Dict keys will be used as values in
            ``meta_col`` (see ``meta_col``).
        meta_col
            Name of a new column to be added to ``obs`` table of the merged
            ExprDataSet object, which can be used for distinguishing cells from
            each dataset. If not specified, no such column will be added.
        merge_uns_slots
            Assuming variable subsets, e.g. most informative genes,
            are stored in the ``uns`` slot, this parameter specifies the
            variable subsets to be merged.
            Note that ``uns`` slots not specified here will be discarded.

        Returns
        -------
        merged_dataset
            Merged dataset.
        """
        dataset_dict = collections.OrderedDict(dataset_dict)

        var_name_list = [dataset.var_names for dataset in dataset_dict.values()]
        var_union = functools.reduce(np.union1d, var_name_list)
        var_intersect = functools.reduce(np.intersect1d, var_name_list)

        for item in dataset_dict:
            dataset_dict[item] = dataset_dict[item].copy(deep=True)[
                :, var_union
            ]  # Avoid contaminating original datasets

        utils.logger.info("Merging uns slots...")
        if merge_uns_slots is None:
            merge_uns_slots = []
        merged_slot = {}
        for slot in merge_uns_slots:
            merged_slot[slot] = []
            for dataset in dataset_dict.values():
                merged_slot[slot].append(dataset.uns[slot])
            merged_slot[slot] = np.intersect1d(
                functools.reduce(np.union1d, merged_slot[slot]), var_intersect)

        utils.logger.info("Merging var data frame...")
        merged_var = []
        for item in dataset_dict:
            var = dataset_dict[item].var.reindex(var_union)
            var.columns = ["_".join([c, item]) for c in var.columns]
            merged_var.append(var)
        merged_var = pd.concat(merged_var, axis=1)

        utils.logger.info("Merging obs data frame...")
        merged_obs = []
        for key in dataset_dict.keys():
            if meta_col:
                dataset_dict[key].obs[meta_col] = key
            merged_obs.append(dataset_dict[key].obs)
        merged_obs = pd.concat(merged_obs, sort=True)

        utils.logger.info("Merging expression matrix...")
        if np.any([
                scipy.sparse.issparse(dataset.exprs)
                for dataset in dataset_dict.values()
        ]):
            merged_exprs = scipy.sparse.vstack([
                scipy.sparse.csr_matrix(dataset.exprs)
                for dataset in dataset_dict.values()
            ])
        else:
            merged_exprs = np.concatenate([
                dataset.exprs for dataset in dataset_dict.values()
            ], axis=0)

        return cls(
            exprs=merged_exprs, obs=merged_obs, var=merged_var,
            uns=merged_slot
        )

    def _prepare_latent_visualization(
            self, method: str, random_seed: int = config._USE_GLOBAL,
            reuse: bool = True, **kwargs
    ) -> None:
        import sklearn.manifold
        import umap

        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed

        # method in {"tSNE", "UMAP"}
        if not reuse:
            for col in self.obs.columns:
                if col.startswith(method):
                    del self.obs[col]
        mask = np.vectorize(lambda x: x.startswith(method))(self.obs.columns)
        if not np.any(mask):
            if method == "tSNE":
                mapper = sklearn.manifold.TSNE(random_state=random_seed, **kwargs)
            elif method == "UMAP":
                mapper = umap.UMAP(random_state=random_seed, **kwargs)
            else:
                raise ValueError("Unknown method!")
            utils.logger.info("Computing %s...", method)
            coord = mapper.fit_transform(self.latent)
            columns = np.vectorize(
                lambda x, method=method: f"{method}{x}"
            )(np.arange(coord.shape[1]) + 1)
            coord_df = pd.DataFrame(coord, index=self.obs_names, columns=columns)
            self.obs = pd.concat([self.obs, coord_df], axis=1)
        else:
            utils.logger.info("Using cached %s...", method)

    def visualize_latent(
            self, hue: typing.Optional[str] = None,
            style: typing.Optional[str] = None,
            method: str = "tSNE", reuse: bool = True, shuffle: bool = True,
            sort: bool = False, ascending: bool = True, size: float = 3.0,
            width: float = 7.0, height: float = 7.0,
            random_seed: int = config._USE_GLOBAL,
            ax: typing.Optional[matplotlib.axes.Axes] = None,
            dr_kws: typing.Optional[typing.Mapping] = None,
            scatter_kws: typing.Optional[typing.Mapping] = None
    ) -> matplotlib.axes.Axes:
        r"""
        Visualize latent space

        Parameters
        ----------
        hue
            Specifies a column in the ``obs`` table or a gene name to use as
            hue of the data points.
        style
            Specifies a column in the ``obs`` table to use as style of data
            points.
        method
            Should be among {"tSNE", "UMAP", None}.
            Specifies the dimension reduction algorithm for visualization.
            If ``None`` is specified, the first two latent
            dimensions will be used for visualization.
        reuse
            Whether to reuse existing visualization coordinates.
        shuffle
            Whether to shuffle data points before plotting.
        sort
            Whether to sort points according to ``hue`` before plotting.
            If set to true, ``shuffle`` takes no effect.
        ascending
            Whether sorting is in the ascending order.
            Only effective when ``sort`` is set to true.
        size
            Point size.
        width
            Figure width.
        height
            Figure height.
        random_seed
            Random seed used in dimension reduction algorithm. If not specified,
            :data:`config.RANDOM_SEED` will be used, which defaults
            to None.
        ax
            Specifies an existing axes to plot onto. If specified,
            ``width`` and ``height`` take no effect.
        dr_kws
            Keyword arguments passed to the dimension reduction algorithm,
            according to ``method``.
            If ``method`` is "tSNE", will be passed to :class:`sklearn.manifold.TSNE`.
            If ``method`` is "UMAP", will be passed to :class:`umap.UMAP`.
        scatter_kws
            Keyword arguments to be passed to :func:`sns.scatterplot`.

        Returns
        -------
        ax
            Visualization plot.
        """
        if dr_kws is None:
            dr_kws = dict()
        if scatter_kws is None:
            scatter_kws = dict()

        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed

        if method is not None:
            self._prepare_latent_visualization(
                method, random_seed=random_seed, reuse=reuse, **dr_kws)
        else:
            method = "latent_"
        if ax is None:
            _, ax = plt.subplots(figsize=(width, height))
        fetch = [f"{method}1", f"{method}2"]
        if hue is not None:
            fetch.append(hue)
        if style is not None:
            fetch.append(style)
        df = self.get_meta_or_var(fetch, normalize_var=True, log_var=True)
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed)
        if hue is not None and sort:
            df = df.sort_values(hue, ascending=ascending)
        ax = sns.scatterplot(
            x=f"{method}1", y=f"{method}2",
            hue=hue, style=style, s=size, data=df, edgecolor=None, ax=ax,
            **scatter_kws
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        if hue is not None:
            _ = ax.legend(
                bbox_to_anchor=(1.05, 0.5), loc="center left",
                borderaxespad=0.0, frameon=False
            )
        return ax

    def obs_correlation_heatmap(
            self, group: typing.Optional[str] = None,
            used_vars: typing.Optional[typing.List[str]] = None,
            cluster_method: str = "complete",
            width: float = 10.0, height: float = 10.0, **kwargs
    ) -> sns.matrix.ClusterGrid:
        r"""
        Correlation heatmap of each observation.

        Parameters
        ----------
        group
            Specifies a column in the ``obs`` table which will be used to label
            rows and columns.
        used_vars
            Specifies variables used to compute correlation. If not specified,
            meaning all variables will be used.
        cluster_method
            Clustering method. See :func:``scipy.cluster.hierarchy.linkage``
            for available options.
        width
            Figure width.
        height
            Figure height.
        kwargs
            Additional keyword arguments will be passed to
            :func:`sns.clustermap`.

        Returns
        -------
        grid
            Visualization plot.
        """
        import matplotlib.patches as mpatches
        import scipy.cluster

        dataset = self if used_vars is None else self[:, used_vars]
        exprs = utils.densify(dataset.exprs)
        exprs = np.log1p(exprs)
        corr = pd.DataFrame(
            np.corrcoef(exprs),
            index=dataset.obs_names, columns=dataset.obs_names
        )
        linkage = scipy.cluster.hierarchy.linkage(
            exprs, method=cluster_method, metric="correlation")
        if group is not None:
            label = self.obs[group]
            label_uniq = np.unique(label.values)
            lut = collections.OrderedDict(zip(label_uniq, sns.hls_palette(
                label_uniq.size, l=0.5, s=0.7)))
            legend_patch = [mpatches.Patch(color=lut[k], label=k) for k in lut]
            label = label.map(lut)
        else:
            legend_patch = None
            label = None
        grid = sns.clustermap(
            corr, row_linkage=linkage, col_linkage=linkage,
            row_colors=label, col_colors=label, figsize=(width, height),
            xticklabels=False, yticklabels=False, **kwargs
        )
        if legend_patch is not None:
            _ = grid.ax_heatmap.legend(
                loc="center left", bbox_to_anchor=(1.05, 0.5),
                handles=legend_patch, frameon=False
            )
        return grid

    def violin(
            self, group: str, var: str, normalize_var: bool = True,
            width: float = 7, height: float = 7,
            ax: typing.Optional[matplotlib.axes.Axes] = None,
            strip_kws: typing.Optional[typing.Mapping] = None,
            violin_kws: typing.Optional[typing.Mapping] = None
    ) -> matplotlib.axes.Axes:
        r"""
        Violin plot across obs groups.

        Parameters
        ----------
        group
            Specifies a column in the ``obs`` table used for cell grouping.
        var
            Variable name.
        normalize_var
            Whether to perform cell normalization.
        width
            Figure width.
        height
            Figure height.
        ax
            Specifies an existing axes to plot onto. If specified,
            ``width`` and ``height`` take no effect.
        strip_kws
            Additional keyword arguments will be passed to :func:`sns.stripplot`.
        violin_kws
            Additional keyword arguments will be passed to :func:`sns.violinplot`.

        Returns
        -------
        ax
            Visualization figure.
        """
        strip_kws = strip_kws or {}
        violin_kws = violin_kws or {}

        df = self.get_meta_or_var(
            [group, var],
            normalize_var=normalize_var, log_var=True
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(width, height))
        ax = sns.stripplot(
            x=group, y=var, data=df,
            color=".3", edgecolor=None, size=3, ax=ax, **strip_kws
        )
        ax = sns.violinplot(
            x=group, y=var, data=df,
            scale="width", ax=ax, inner=None, **violin_kws
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax

    def annotation_confidence(
            self, annotation: typing.Union[str, typing.List[str]],
            used_vars: typing.Optional[typing.Union[str, typing.List[str]]] = None,
            metric: str = "cosine", return_group_percentile: bool = True
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute annotation confidence of each obs (cell) based on
        sample silhouette score.

        Parameters
        ----------
        annotation
            Specifies annotation for which confidence will be computed.
            If passed an array-like, it should be 1 dimensional with length
            equal to obs number, and will be used directly as annotation.
            If passed a string, it should be a column name in ``obs``.
        used_vars
            Specifies the variable set used to evaluate ``metric``,
            If not specified, all variables are used. If given a string,
            it should be a slot in `uns`. If given a 1-d array, it should
            contain variable names to be used.
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
            annotation = self.obs[annotation].values
        annotation = utils.encode_integer(annotation)[0]
        if used_vars is None:
            used_vars = self.var_names
        elif isinstance(used_vars, str):
            used_vars = self.uns[used_vars]
        exprs = self[:, used_vars].exprs
        exprs = exprs.log1p() if scipy.sparse.issparse(exprs) \
            else np.log1p(exprs)
        confidence = sklearn.metrics.silhouette_samples(
            exprs, annotation, metric=metric)
        if return_group_percentile:
            normalized_confidence = np.zeros_like(confidence)
            for l in np.unique(annotation):
                mask = annotation == l
                normalized_confidence[mask] = (
                    scipy.stats.rankdata(confidence[mask]) - 1
                ) / (mask.sum() - 1)
            return confidence, normalized_confidence
        return confidence

    def fast_markers(
            self, group: str,
            used_genes: typing.Optional[typing.List[str]] = None,
            alternative: str = "two-sided", multitest: str = "bonferroni",
            min_pct: float = 0.1, min_pct_diff: float = -np.inf,
            logfc_threshold: float = 0.25, pseudocount: float = 1.0,
            n_jobs: int = 1
    ) -> typing.Mapping[str, pd.DataFrame]:
        r"""
        Find markers for each group by one-vs-rest Wilcoxon rank sum test.
        This is a fast implementation of the ``FindAllMarkers`` function
        in Seurat v2.

        Parameters
        ----------
        group
            Specifies a column in ``obs`` that determines cell grouping.
        used_genes
            A sequence of genes in which to search for markers.
        alternative
            Alternative hypothesis, should be among
            {"two-sided", "greater", "less"}.
        multitest
            Method of multiple test p-value correction. Check
            :func:`statsmodels.stats.multitest.multipletests` for available
            options.
        min_pct
            Minimal percent of cell expressing gene of interest, either in
            group or rest, for it to be considered in statistical test.
        min_pct_diff
            Minimal percent difference of cell expressing gene of interest
            in group and rest, for it to be considered in statistical test.
        logfc_threshold
            Minimal log fold change in average expression level of gene
            of interest, between group and rest, for it to be considered in
            statistical test.
        pseudocount
            Pseudocount to be added when computing log fold change.
        n_jobs
            Number of parallel running threads to use.

        Returns
        -------
        summary
            Each element, named by cell group, is a table containing
            differential expression results.
            Columns of each table are:
            "pct_1": percent of cells expressing the gene in group of interest
            "pct_2": percent of cells expressing the gene in rest
            "logfc": log fold change of mean expression between group and rest
            "stat": statistic of Wilcoxon rank sum test
            "z": normal approximation statistic of Wilcoxon rank sum test
            "pval": p-value of Wilcoxon rank sum test
            "padj": p-value adjusted for multiple test
        """
        import statsmodels.stats.multitest

        if used_genes is None:
            used_genes = self.var_names
        mat = self[:, used_genes].exprs.toarray()
        group_indices, labels = utils.encode_integer(self.obs[group].values)
        group_onehot = utils.encode_onehot(group_indices).astype(bool).toarray()
        n_x = np.sum(group_onehot, axis=0).astype(np.float64)
        n_y = group_onehot.shape[0] - n_x
        n_xy_prod = n_x * n_y
        n_xy_plus = n_x + n_y

        def ranksum_thread(vec):
            r"""
            Wilcoxon rank sum test for one feature
            Adapted from R functions:
                ``Seurat::FindMarkers`` and ``stats::wilcox.test``
            """

            # Preparation
            vec = vec.toarray().ravel() if scipy.sparse.issparse(vec) else vec
            pct = np.empty((group_onehot.shape[1], 2))
            logfc = np.empty((group_onehot.shape[1]))

            for i in range(group_onehot.shape[1]):
                mask = group_onehot[:, i].ravel()
                pct[i, 0] = round(np.sum(vec[mask] > 0) / n_x[i], 3)
                pct[i, 1] = round(np.sum(vec[~mask] > 0) / n_y[i], 3)
                logfc[i] = np.log(vec[mask].mean() + pseudocount) - \
                    np.log(vec[~mask].mean() + pseudocount)

            # Percent expressed filtering
            pct_max = pct.max(axis=1)
            pct_min = pct.min(axis=1)
            pct_diff = pct_max - pct_min
            pct_mask = (pct_max > min_pct) & (pct_diff > min_pct_diff)

            # Fold change filtering
            if alternative == "greater":
                logfc_mask = logfc > logfc_threshold
            elif alternative == "less":
                logfc_mask = logfc < -logfc_threshold
            elif alternative == "two-sided":
                logfc_mask = abs(logfc) > logfc_threshold

            total_mask = pct_mask & logfc_mask
            if not np.any(total_mask):
                nan_placeholder = np.empty(group_onehot.shape[1])
                nan_placeholder[:] = np.nan
                return pct[:, 0].ravel(), pct[:, 1].ravel(), logfc, \
                    nan_placeholder, nan_placeholder, nan_placeholder

            # Rank sum test
            rank = scipy.stats.rankdata(vec)
            n_ties = np.unique(rank, return_counts=True)[1]

            stat = np.empty(group_onehot.shape[1])
            for i in range(group_onehot.shape[1]):
                mask = group_onehot[:, i].ravel()
                if total_mask[i]:
                    stat[i] = rank[mask].sum() - n_x[i] * (n_x[i] + 1) / 2
                else:
                    stat[i] = np.nan
            z = stat - n_x * n_y / 2
            sigma = np.sqrt((n_xy_prod / 12) * (
                (n_xy_plus + 1) -
                (n_ties ** 3 - n_ties).sum() / (n_xy_plus * (n_xy_plus - 1))
            ))
            if alternative == "two-sided":
                correction = np.sign(z) * 0.5
            elif alternative == "greater":
                correction = 0.5
            elif alternative == "less":
                correction = -0.5
            z = (z - correction) / sigma
            if alternative == "two-sided":
                pval = 2 * np.stack([
                    scipy.stats.norm.sf(z),
                    scipy.stats.norm.cdf(z)
                ], axis=0).min(axis=0)
            elif alternative == "greater":
                pval = scipy.stats.norm.sf(z)
            elif alternative == "less":
                pval = scipy.stats.norm.cdf(z)
            return pct[:, 0].ravel(), pct[:, 1].ravel(), logfc, stat, z, pval

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            result = executor.map(ranksum_thread, mat.T)

        pct_1_list, pct_2_list, logfc_list, stat_list, z_list, pval_list = \
            [], [], [], [], [], []
        for pct_1, pct_2, logfc, stat, z, pval in result:
            pct_1_list.append(pct_1)
            pct_2_list.append(pct_2)
            logfc_list.append(logfc)
            stat_list.append(stat)
            z_list.append(z)
            pval_list.append(pval)
        pct_1 = np.stack(pct_1_list, axis=0).T
        pct_2 = np.stack(pct_2_list, axis=0).T
        logfc = np.stack(logfc_list, axis=0).T
        stat = np.stack(stat_list, axis=0).T
        z = np.stack(z_list, axis=0).T
        pval = np.stack(pval_list, axis=0).T

        def my_multitest(pval):
            pval = pval.copy()
            mask = ~np.isnan(pval)
            if np.any(mask):
                pval[mask] = statsmodels.stats.multitest.multipletests(
                    pval[mask], method=multitest
                )[1]
            return pval
        padj = np.apply_along_axis(my_multitest, 1, pval)

        summary = {}
        for i in range(group_onehot.shape[1]):
            summary[labels[i]] = pd.DataFrame({
                "pct_1": pct_1[i],
                "pct_2": pct_2[i],
                "logfc": logfc[i],
                "stat": stat[i],
                "z": z[i],
                "pval": pval[i],
                "padj": padj[i]
            }, index=used_genes).sort_values(by=["z"], ascending=False)
        return summary

    def to_anndata(self) -> anndata.AnnData:
        r"""
        Convert a :class:`ExprDataSet` object to an
        :class:`anndata.AnnData` object.

        Returns
        -------
        dataset
            Resulting :class:`anndata.AnnData` object.
        """
        return anndata.AnnData(
            X=self.exprs, obs=self.obs, var=self.var, uns=dict(self.uns))

    @classmethod
    def from_anndata(cls, ad: anndata.AnnData) -> "ExprDataSet":
        r"""
        Create a :class:`ExprDataSet` object from an existing
        :class:`anndata.AnnData` object.

        Parameters
        ----------
        ad
            An existing :class:`anndata.AnnData` object.

        Returns
        -------
        dataset
            Resulting :class:`ExprDataSet` object.
        """
        return cls(ad.X, ad.obs, ad.var, ad.uns)

    def to_loom(self, file: str) -> loompy.loompy.LoomConnection:
        r"""
        Convert a :class:`ExprDataSet` object to a
        :class:`loompy.loompy.LoomConnection` object. Note that data will be
        written to a loom file specified by ``file`` in this process.

        Parameters
        ----------
        file
            Specifies the loom file to be written

        Returns
        -------
        lm
            Resulting connection to the loom file.
        """
        assert "var_name" not in self.var.columns and \
            "obs_name" not in self.obs.columns
        loompy.create(file, self.exprs.T, {**{
            key: np.array(val)
            for key, val in self.var.to_dict("list").items()
        }, "var_name": np.array(self.var.index)}, {**{
            key: np.array(val)
            for key, val in self.obs.to_dict("list").items()
        }, "obs_name": np.array(self.obs.index)})
        return loompy.connect(file)

    @classmethod
    def from_loom(cls, lm: loompy.loompy.LoomConnection) -> "ExprDataSet":
        r"""
        Create a :class:`ExprDataSet` object from an existing
        :class:`loompy.loompy.LoomConnection` object.

        Parameters
        ----------
        lm
            An existing :class:`loompy.loompy.LoomConnection` object.

        Returns
        -------
        dataset
            Resulting :class:`ExprDataSet` object.
        """
        return cls(
            lm[:, :].T,
            pd.DataFrame({
                key: val for key, val in lm.ca.items()
                if key != "obs_name"
            }, index=lm.ca["obs_name"] if "obs_name" in lm.ca.keys() else None),
            pd.DataFrame({
                key: val for key, val in lm.ra.items()
                if key != "var_name"
            }, index=lm.ra["var_name"] if "var_name" in lm.ra.keys() else None),
            {}
        )

    def write_table(self, filename: str, orientation: str = "cg", **kwargs) -> None:
        r"""
        Write the expression matrix to a plain-text file.
        Note that ``obs`` (cell) meta table, ``var`` (gene) meta table and data
        in the ``uns`` slot are discarded, only the expression matrix is written
        to the file.

        Parameters
        ----------
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
                utils.densify(self.exprs),
                index=self.obs_names,
                columns=self.var_names
            )
        elif orientation == "gc":
            df = pd.DataFrame(
                utils.densify(self.exprs.T),
                index=self.var_names,
                columns=self.obs_names
            )
        else:  # pragma: no cover
            raise ValueError("Invalid orientation!")
        df.to_csv(filename, **kwargs)

    @classmethod
    def read_table(
            cls, filename: str, orientation: str = "cg",
            sparsify: bool = False, **kwargs
    ) -> "ExprDataSet":
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
            An :class:`ExprDataSet` object loaded from the file.
        """
        df = pd.read_csv(filename, **kwargs)
        if orientation == "gc":
            df = df.T
        return cls(
            scipy.sparse.csr_matrix(df.values) if sparsify else df.values,
            pd.DataFrame(index=df.index),
            pd.DataFrame(index=df.columns),
            {}
        )


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


def write_clean(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.dtype.type is np.object_:
        data[utils.isnan(data)] = config._NAN_REPLACEMENT
    if data.dtype.type in (np.str_, np.object_):
        data = utils.encode(data)
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    # d = utils.dotdict()
    d = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def dict_to_group(d, group):
    for key in d:
        if isinstance(d[key], dict):
            dict_to_group(d[key], group.create_group(key))
        else:
            try:
                value = write_clean(d[key])
                if value.size == 1:
                    h5opts = config.H5_TRACK_OPTS
                else:
                    h5opts = {**config.H5_COMPRESS_OPTS, **config.H5_TRACK_OPTS}
                group.create_dataset(key, data=value, **h5opts)
            except Exception:  # pylint: disable=broad-except
                utils.logger.warning("Slot %s failed to save!", key)


def df_to_dict(df):
    d = collections.OrderedDict()
    for column in df.columns:
        d[column] = df[column].values
    return d


def check_hybrid_path(hybrid_path):
    file_name, h5_path = hybrid_path.split("//")
    if not os.path.exists(file_name):
        return False
    with h5py.File(file_name, "r") as f:
        return h5_path in f


def read_hybrid_path(hybrid_path):
    file_name, h5_path = hybrid_path.split("//")
    with h5py.File(file_name, "r") as f:
        if isinstance(f[h5_path], h5py.Group):
            fetch = dict_from_group(f[h5_path])
        else:
            fetch = read_clean(f[h5_path][...])
    return fetch


def write_hybrid_path(obj, hybrid_path):
    file_name, h5_path = hybrid_path.split("//")
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with h5py.File(file_name, "a") as f:
        if h5_path in f:
            del f[h5_path]
        if isinstance(obj, dict):
            dict_to_group(obj, f.create_group(h5_path))
        else:
            obj = write_clean(obj)
            if obj.size == 1:
                h5opts = config.H5_TRACK_OPTS
            else:
                h5opts = {**config.H5_COMPRESS_OPTS, **config.H5_TRACK_OPTS}
            f.create_dataset(h5_path, data=obj, **h5opts)
