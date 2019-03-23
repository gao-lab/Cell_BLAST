"""
Dataset utilities
"""

import os
import copy
import collections
import functools
import concurrent.futures

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import h5py
import sklearn.metrics


from . import message
from . import utils
from . import config


class ExprDataSet(object):

    """
    Main data class, which is based on the data structure of ``AnnData``.
    Note that in this package we restrict to scRNA-seq data, so the stored
    matrix is always the expression matrix, and the terms "obs" and "var" are
    used interchangeably with "cell" and "gene".

    Parameters
    ----------
    exprs : numpy.ndarray, scipy.sparse.spmatrix
        A :math:`obs \\times var` expression matrix in the form of
        either numpy array or scipy sparse matrix.
    obs : pandas.DataFrame
        Cell meta table, each row corresponding to a row in ``exprs``.
    var : pandas.DataFrame
        Gene meta table, each row corresponding to a column in ``exprs``.
    uns : dict
        Unstructured meta information, e.g. list of highly variable genes.
        Values should be numpy arrays if they are to be saved to file.

    Examples
    --------

    An ``ExprDataSet`` object can be constructed from an expression matrix, an
    observation(cell) meta table, a variable(gene) meta table, and some
    unstructured data:

    >>> data_obj = Cell_BLAST.data.ExprDataSet(exprs, obs, var, uns)

    If you have an ``AnnData`` object, you can directly convert it
    to an ``ExprDataSet`` object:

    >>> data_obj = Cell_BLAST.data.ExprDataSet.from_anndata(anndata_obj)

    It's also possible in the opposite direction:

    >>> anndata_obj = data_obj.to_anndata()

    ``ExprDataSet`` objects support many forms of slicing, including
    python slicing, obs/var name matching, numeric indexing and boolean mask:

    >>> subdata_obj = data_obj[0:10, np.arange(10)]
    >>> subdata_obj = data_obj[
    ...     data_obj.obs["cell_ontology_class"] == "endothelial",
    ...     ["gene_1", "gene_2"]
    ... ]

    Note that for variable name matching, if a variable does not exist in the
    original dataset, it will be filled with zeros in the returned dataset,
    with a warning message.

    They also support easy saving and loading:

    >>> data_obj.save("data.h5")
    >>> data_obj = Cell_BLAST.data.ExprDataSet.load("data.h5")

    Some utilities used in the Cell_BLAST/DIRECTi pipeline are also supported,
    including but not limited to:

    Dataset merging

    >>> combined_data_obj = Cell_BLAST.data.ExprDataSet.merge_datasets({
    ...     "data1": data_obj1,
    ...     "data2": data_obj2,
    ...     "data3": data_obj3
    ... })

    Data visualization

    >>> data_obj.latent = latent_matrix
    >>> _ = data_obj.visualize_latent("cell_type")
    >>> _ = data_obj.violin("cell_type", "gene_name")
    >>> _ = data_obj.obs_correlation_heatmap()

    Find markers:

    >>> marker_dict = data_obj.fast_markers("cell_type")

    Computation of annotation confidence

    >>> confidence = data_obj.annotation_confidence("cell_type")
    """

    def __init__(self, exprs, obs, var, uns):
        # TODO: uns slots that are not numpy arrays may have trouble saving
        assert exprs.shape[0] == obs.shape[0] and exprs.shape[1] == var.shape[0]
        if scipy.sparse.issparse(exprs):
            self.exprs = exprs.tocsr()
        else:
            self.exprs = exprs
        self.obs = obs
        self.var = var
        self.uns = utils.dotdict(uns)

    @property
    def X(self):  # For compatibility with `AnnData`
        """
        :math:`obs \\times var` matrix, same as ``exprs``
        """
        return self.exprs

    @property
    def obs_names(self):
        """
        Name of observations (cells)
        """
        return self.obs.index

    @obs_names.setter
    def obs_names(self, new_names):
        assert len(new_names) == self.obs.shape[0]
        self.obs.index = new_names

    @property
    def var_names(self):
        """
        Name of variables (genes)
        """
        return self.var.index

    @var_names.setter
    def var_names(self, new_names):
        assert len(new_names) == self.var.shape[0]
        self.var.index = new_names

    @property
    def shape(self):
        """
        Shape of dataset (:math:`obs \\times var`)
        """
        return self.exprs.shape

    @property
    def latent(self):
        """
        Latent space coordinate
        """
        mask = np.vectorize(lambda x: x.startswith("latent_"))(self.obs.columns)
        if np.any(mask):
            return self.obs.loc[:, mask].values
        else:
            raise ValueError("No latent has been registered!")

    @latent.setter
    def latent(self, latent):
        for col in self.obs.columns:  # Remove previous result
            if col.startswith("latent_"):
                del self.obs[col]
            if col.startswith("tSNE"):
                del self.obs[col]
            if col.startswith("UMAP"):
                del self.obs[col]
        assert latent.shape[0] == self.shape[0]
        columns = np.vectorize(
            lambda x: "latent_%d" % x
        )(np.arange(latent.shape[1]) + 1)
        latent_df = pd.DataFrame(latent, index=self.obs_names, columns=columns)
        self.obs = pd.concat([self.obs, latent_df], axis=1)

    def normalize(self, target=10000):
        """
        Obs-wise (cell-wise) normalization if the matrix.
        Note that only the matrix gets copied in the returned dataset, but meta
        tables are only references to the original dataset.

        Parameters
        ----------
        target : int
            Target value of normalization, by default 10000.

        Returns
        -------
        normalized : ExprDataSet
            Normalized ExprDataSet object.
        """
        import sklearn.preprocessing
        tmp = self.copy()
        tmp.exprs = sklearn.preprocessing.normalize(
            tmp.exprs, norm="l1", copy=True
        ) * target
        return tmp

    def __getitem__(self, slices):
        """
        Support 2-d slicing by integer index, boolean mask and also name,
        """
        if len(slices) == 2:
            obs_slice, var_slice = slices
            verbose = 1
        elif len(slices) == 3:
            obs_slice, var_slice, verbose = slices
        else:  # pragma: no cover
            raise ValueError("Invalid slicing!")

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
            if new_var_names.size > 0 and verbose > 0:  # pragma: no cover
                message.warning(
                    "%d out of %d variables are not found, will be set to zero!" %
                    (len(new_var_names), len(var_slice))
                )
                if verbose > 1:
                    print(str(new_var_names.tolist()).strip("[]"))
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

        return ExprDataSet(
            exprs=exprs.copy(), obs=obs, var=var,
            uns=copy.deepcopy(dict(self.uns))
        )

    def get_meta_or_var(self, names, normalize_var=False, log_var=False):
        """
        Get either meta information (column names in ``obs``) or
        variable values (row names in ``var``).

        Parameters
        ----------
        names : list
            List of names that specifies meta information / variables
            to be fetched.
        normalize_var : bool
            Whether to do cell-normalization before fetching variable values,
            by default False.
        log_var : bool
            Whether to apply log transform for fetched variable values,
            by default False.

        Returns
        -------
        fetched : pandas.DataFrame
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

    def copy(self, deep=False):
        """
        Produce a copy of the dataset.

        Parameters
        ----------
        deep : bool
            Whether to perform deep copy, by default False.

        Returns
        -------
        copied : ExprDataSet
            Copy of the dataset.
        """
        if deep:
            return ExprDataSet(
                self.exprs.copy(), self.obs.copy(), self.var.copy(),
                copy.deepcopy(dict(self.uns))  # dotdict can't be deep copied
            )
        return ExprDataSet(self.exprs, self.obs, self.var, self.uns)

    def write_dataset(self, filename):
        """
        Write the dataset to a file.

        Parameters
        ----------
        filename : str
            File to be written (content in hdf5 format).
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with h5py.File(filename, "w") as f:
            if scipy.sparse.issparse(self.exprs):
                self.exprs.sort_indices()  # Compatibility with R
                g = f.create_group("exprs")
                g.create_dataset("data", data=self.exprs.data, **config.H5OPTS)
                g.create_dataset("indices", data=self.exprs.indices, **config.H5OPTS)
                g.create_dataset("indptr", data=self.exprs.indptr, **config.H5OPTS)
                g.create_dataset("shape", data=self.exprs.shape, **config.H5OPTS)
            else:
                f.create_dataset("exprs", data=self.exprs, **config.H5OPTS)
            f.create_dataset("obs_names", data=utils.encode(self.obs_names.values), **config.H5OPTS)
            f.create_dataset("var_names", data=utils.encode(self.var_names.values), **config.H5OPTS)
            dict_to_group(df_to_dict(self.obs), f.create_group("obs"))
            dict_to_group(df_to_dict(self.var), f.create_group("var"))
            dict_to_group(self.uns, f.create_group("uns"))

    @classmethod
    def read_dataset(cls, filename, sparsify=False, skip_exprs=False):
        """
        Read dataset from file (saved by ``write_dataset``).

        Parameters
        ----------
        filename : str
            File to read from (content in hdf5 format).
        sparsify : bool
            Whether to convert the expression matrix into sparse format,
            by default False.
        skip_exprs : bool
            Whether to skip reading the expression matrix and use all zeros,
            by default False. This option is provided to accelerate data
            reading if only meta information are needed.

        Returns
        -------
        loaded_dataset : ExprDataSet
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

    def map_vars(self, mapping, map_uns_slots=None, verbose=1):
        """
        Map variables of the dataset to some other terms,
        e.g. gene ortholog groups.

        Parameters
        ----------
        mapping : pandas.DataFrame
            A 2-column data frame defining variable name mapping. First column
            is source variable name and second column is target variable name.
        map_uns_slots : list
            Assuming variable subsets, e.g. Seurat variable genes,
            are stored in the ``uns`` slot, this parameter specifies which slots
            in ``uns`` should also be mapped, by default None.
            Note that ``uns`` slots not included will be left as is.
        verbose : {0, 1, 2}
            If ``verbose=0``, no warning message will be printed.
            If ``verbose=1``, the number of source/target items that are
            ambiguously mapped will be reported.
            If ``verbose=2``, a list of such ambiguous vars will be reported.
            Default value is 1.

        Returns
        -------
        mapped : Cell_BLAST.data.ExprDataSet
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
        if verbose > 0 and amb_src_mask.sum() > 0:  # pragma: no cover
            message.warning("%d ambiguous source items found!" %
                            amb_src_mask.sum())
            if verbose > 1:
                print(source[amb_src_mask].tolist())
        if verbose > 0 and amb_tgt_mask.sum() > 0:  # pragma: no cover
            message.warning("%d ambiguous target items found!" %
                            amb_tgt_mask.sum())
            if verbose > 1:
                print(target[amb_tgt_mask].tolist())

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
    def merge_datasets(cls, dataset_dict, meta_col=None,
                       merge_uns_slots=None, verbose=1):
        """
        Merge multiple dataset objects into a meta dataset.

        Parameters
        ----------
        dataset_dict : dict
            A dict of ExprDataSet objects. Dict keys will be used as values in
            ``meta_col`` (see ``meta_col``).
        meta_col : str
            Name of the new column to be added in ``obs`` slot of the merged
            ExprDataSet object, used for distinguishing each dataset,
            by default None, meaning that no such column will be added.
        merge_uns_slots : list
            Assuming variable subsets, e.g. Seurat variable genes, are stored
            in the ``uns`` slot, this parameter specifies variable subsets to be
            merged, by default None.
            Note that uns slots not specified will be discarded.
        verbose : {0, 1, 2}
            If ``verbose=0``, no warning message will be printed.
            If ``verbose=1``, the number of vars in the var union that's
            missing in each dataset will be reported.
            If ``verbose=2``, a list of such missing vars in each dataset
            will be reported.

        Returns
        -------
        merged_dataset : ExprDataSet
            Merged dataset.
        """
        dataset_dict = collections.OrderedDict(dataset_dict)

        var_name_list = [dataset.var_names for dataset in dataset_dict.values()]
        var_union = functools.reduce(np.union1d, var_name_list)
        var_intersect = functools.reduce(np.intersect1d, var_name_list)

        for item in dataset_dict:
            dataset_dict[item] = dataset_dict[item].copy(deep=True)[
                :, var_union, verbose
            ]  # Avoid contaminating original datasets

        if verbose > 0:
            message.info("Merging uns slots...")
        if merge_uns_slots is None:
            merge_uns_slots = []
        merged_slot = {}
        for slot in merge_uns_slots:
            merged_slot[slot] = []
            for dataset in dataset_dict.values():
                merged_slot[slot].append(dataset.uns[slot])
            merged_slot[slot] = np.intersect1d(
                functools.reduce(np.union1d, merged_slot[slot]), var_intersect)

        if verbose > 0:
            message.info("Merging var data frame...")
        merged_var = []
        for item in dataset_dict:
            var = dataset_dict[item].var.reindex(var_union)
            var.columns = ["_".join([c, item]) for c in var.columns]
            merged_var.append(var)
        merged_var = pd.concat(merged_var, axis=1)

        if verbose > 0:
            message.info("Merging obs data frame...")
        merged_obs = []
        for key in dataset_dict.keys():
            if meta_col:
                dataset_dict[key].obs[meta_col] = key
            merged_obs.append(dataset_dict[key].obs)
        merged_obs = pd.concat(merged_obs)

        if verbose > 0:
            message.info("Merging expression matrix...")
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
        self, method, random_seed=config._USE_GLOBAL, reuse=True, **kwargs
    ):
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
            message.info("Computing %s..." % method)
            coord = mapper.fit_transform(self.latent)
            columns = np.vectorize(
                lambda x, method=method: "%s%d" % (method, x)
            )(np.arange(coord.shape[1]) + 1)
            coord_df = pd.DataFrame(coord, index=self.obs_names, columns=columns)
            self.obs = pd.concat([self.obs, coord_df], axis=1)
        else:
            message.info("Using cached %s..." % method)

    def visualize_latent(
        self, hue=None, method="tSNE", reuse=True, shuffle=True, sort=False,
        ascending=True, size=3, width=7, height=7,
        random_seed=config._USE_GLOBAL, ax=None,
        dr_kws=None, scatter_kws=None
    ):
        """
        Visualize latent space

        Parameters
        ----------
        hue : str
            Specify a column in the ``obs`` slot or a row in the ``var`` slot
            to use as color, by default None.
        method : {"tSNE", "UMAP", None}
            Specify the dimension reduction algorithm for visualization,
            by default "tSNE". If ``None`` is specified, the first two latent
            dimensions will be used for visualization.
        reuse : bool
            Whether to reuse existing visualization coordinates,
            by default True.
        shuffle : bool
            Whether to shuffle point before plotting, by default True.
        sort : bool
            Whether to sort points according to ``color`` before plotting,
            by default False. If set to true, ``shuffle`` takes no effect.
        ascending : bool
            Whether sorting is ascending, by default True. Only effective when
            ``sort`` is set to true.
        size : int
            Point size, by default 3.
        width : float
            Figure width, by default 7.
        height : float
            Figure height, by default 7.
        random_seed : int
            Random seed used in dimension reduction algorithm. If not specified,
            ``Cell_BLAST.config.RANDOM_SEED`` will be used, which defaults
            to None.
        ax : matplotlib.axes.Axes
            Specify an existing axes to plot onto, by default None.
            If specified, ``width`` and ``height`` take no effect.
        dr_kws: dict
            Keyword arguments to be passed to the dimension reduction
            algorithm, according to ``method``.
            If ``method`` is "tSNE", will be passed to ``sklearn.manifold.TSNE``.
            If ``method`` is "UMAP", will be passed to ``umap.UMAP``.
        scatter_kws : dict
            Keyword arguments to be passed to ``seaborn.scatterplot``,
            by default None.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Visualization plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

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
        fetch = ["%s1" % method, "%s2" % method]
        if hue is not None:
            fetch.append(hue)
        df = self.get_meta_or_var(fetch, normalize_var=True, log_var=True)
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed)
        if hue is not None and sort:
            df = df.sort_values(hue, ascending=ascending)
        ax = sns.scatterplot(
            x="%s1" % method, y="%s2" % method,
            hue=hue, s=size, data=df, edgecolor=None, ax=ax,
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
        self, group=None, used_vars=None,
        cluster_method="complete", width=10, height=10, **kwargs
    ):
        """
        Correlation heatmap of each observation.

        Parameters
        ----------
        group : str
            Specify a column in ``obs`` which will be used to color rows and
            columns, by default None.
        used_vars : array_like
            Specify variables used to compute correlation, by default None,
            meaning all variables will be used.
        cluster_method : str
            Clustering method, by default "complete". See
            ``scipy.cluster.hierarchy.linkage`` for available options.
        width : float
            Figure width, by default 10.
        height : float
            Figure height, by default 10.
        **kwargs
            Additional keyword arguments will be passed to
            ``seaborn.clustermap``.

        Returns
        -------
        grid : seaborn.matrix.ClusterGrid
            Visualization plot.
        """
        import matplotlib.patches as mpatches
        import seaborn as sns
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
        self, group, var, normalize_var=True, width=7, height=7,
        ax=None, **kwargs
    ):
        """
        Violin plot across obs groups.

        Parameters
        ----------
        group : str
            Specify a column in ``obs`` that provides obs grouping.
        var : str
            Variable name.
        normalize_var : bool
            Whether to perform obs normalization, by default True.
        width : float
            Figure width, by default 10.
        height : float
            Figure height, by default 10.
        ax : matplotlib.axes.Axes
            Specify an existing axes to plot onto, by default None.
            If specified, ``width`` and ``height`` take no effect.
        **kwargs
            Additional keyword arguments will be passed to
            ``seaborn.violinplot``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Visualization figure.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self.get_meta_or_var(
            [group, var],
            normalize_var=normalize_var, log_var=True
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(width, height))
        ax = sns.violinplot(
            x=group, y=var, data=df,
            scale="width", ax=ax, inner="point", **kwargs
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax

    def annotation_confidence(
        self, annotation, used_vars=None, metric="cosine",
        return_group_percentile=True
    ):
        """
        Compute annotation confidence of each obs based on
        sample silhouette score.

        Parameters
        ----------
        annotation : array_like, str
            Specifies annotation for which confidence will be computed.
            If passed an array-like, it should be 1 dimensional with length
            equal to obs number, and will be used directly as annotation.
            If passed a string, it should be a column name in ``obs``.
        used_vars : str or array_like
            Specifies the variable set used to evaluate ``metric``,
            by default None, meaning all variables are used. If given a string,
            it should be a slot in `uns`. If given a 1-d array, it should
            contain variable names to be used.
        metric : str
            Specifies distance metric used to compute sample silhouette scores,
            by default "cosine".
        return_group_percentile : bool
            Whether to return within group confidence percentile, instead of
            raw sample silhouette score, by default True.

        Returns
        -------
        confidence : numpy.ndarray
            1 dimensional numpy array containing annotation confidence for
            each obs.
        group_percentile : numpy.ndarray
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

    # TODO: use numba to further increase speed and achieve full parallel computing
    def fast_markers(self, group, used_genes=None,
                     alternative="two-sided", multitest="bonferroni",
                     min_pct=0.1, min_pct_diff=-np.inf,
                     logfc_threshold=0.25, pseudocount=1,
                     n_jobs=1):
        """
        Find markers for each group by one-vs-rest Wilcoxon rank sum test.
        This is a fast implementation of the ``FindAllMarkers`` function
        in Seurat 2.

        Parameters
        ----------
        group : str
            Specify a column in ``obs`` that determines cell grouping.
        used_genes : array_like
            A sequence of genes in which to search for markers.
        alternative : {"two-sided", "greater", "less"}
            Alternative hypothesis, by default "two-sided".
        multitest : str
            Method of multiple test p-value correction, by default "bonferroni".
            Check ``statsmodels.stats.multitest.multipletests`` for available
            options.
        min_pct : float
            Minimal percent of cell expressing gene of interest, either in
            group or rest, for it to be considered in statistical test,
            by default 0.1.
        min_pct_diff : float
            Minimal percent difference of cell expressing gene of interest
            in group and rest, for it to be considered in statistical test,
            by default -np.inf.
        logfc_threshold : float
            Minimal log fold change in average expression level of gene
            of interest, between group and rest, for it to be considered in
            statistical test, by default 0.25.
        pseudocount : float
            Pseudocount to be added when computing log fold change,
            by default 1.
        n_jobs : int
            Number of parallel running threads to use.

        Returns
        -------
        summary : dict
            Each element, named by cell group, is a pandas DataFrame containing
            differential expression results.
            Columns of each DataFrame are:
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
            """
            Wilcoxon rank sum test for one feature
            Adapted from the following R functions:
                `Seurat::FindMarkers` and `stats::wilcox.test`
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

    def to_anndata(self):
        """
        Convert a ``Cell_BLAST.data.ExprDataSet`` object to an
        ``anndata.AnnData`` object.

        Returns
        -------
        dataset : anndata.AnnData
            Resulting ``anndata.AnnData`` object.
        """
        import anndata
        return anndata.AnnData(
            X=self.exprs, obs=self.obs, var=self.var, uns=self.uns)

    @classmethod
    def from_anndata(cls, ad):
        """
        Create a ``Cell_BLAST.data.ExprDataSet`` object from an existing
        ``anndata.AnnData`` object.

        Parameters
        ----------
        ad : anndata.AnnData
            An existing ``anndata.AnnData`` object.

        Returns
        -------
        dataset : Cell_BLAST.data.ExprDataSet
            Resulting ``Cell_BLAST.data.ExprDataSet`` object.
        """
        return cls(ad.X, ad.obs, ad.var, ad.uns)

    def to_loom(self, file):
        """
        Convert a ``Cell_BLAST.data.ExprDataSet`` object to a
        ``loompy.loompy.LoomConnection`` object. Note that data will be
        written to a loom file specified by ``file`` in this process.

        Parameters
        ----------
        file : str
            Specify the loom file to be written

        Returns
        -------
        lm : loompy.loompy.LoomConnection
            Resulting ``loompy`` connection to the loom file.
        """
        import loompy
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
    def from_loom(cls, lm):
        """
        Create a ``Cell_BLAST.data.ExprDataSet`` object from an existing
        ``loompy.loompy.LoomConnection`` object.

        Parameters
        ----------
        lm : loompy.loompy.LoomConnection
            An existing ``loompy.loompy.LoomConnection`` object.

        Returns
        -------
        dataset : Cell_BLAST.data.ExprDataSet
            Resulting ``Cell_BLAST.data.ExprDataSet`` object.
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

    def write_table(self, filename, orientation="cg", **kwargs):
        """
        Write expression matrix to a text based table file

        Parameters
        ----------
        filename : str
            Name of the file to be written.
        orientation : {"cg", "gc"}
            Specifies whether to write in :math:`obs \\times var` or
            :math:`obs \\times var` orientation.
        **kwargs
            Additional keyword arguments will be passed to
            ``pandas.DataFrame.to_csv``.
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
    def read_table(cls, filename, orientation="cg", sparsify=False, **kwargs):
        """
        Read expression matrix from a text based table file

        Parameters
        ----------
        filename : str
            Name of the file to read from.
        orientation : {"cg", "gc"}
            Specifies whether matrix in the file is in
            :math:`cell \\times gene` or :math:`gene \\times cell` orientation.
        sparsify : bool
            Whether to convert the expression matrix into sparse format.
        **kwargs
            Additional keyword arguments will be passed to ``pandas.read_csv``.

        Returns
        -------
        loaded_dataset : ExprDataSet
            An ExprDataSet object loaded from the file.
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
    if data.size == 1:
        data = data.flat[0]
    return data


def write_clean(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.dtype.type in (np.str_, np.object_):
        data = utils.encode(data)
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
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
            value = write_clean(d[key])
            if value.size == 1:
                group.create_dataset(key, data=value)
            else:
                group.create_dataset(key, data=value, **config.H5OPTS)


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
                f.create_dataset(h5_path, data=obj)
            else:
                f.create_dataset(h5_path, data=obj, **config.H5OPTS)
