"""
Cell BLAST based on DIRECTi models
"""


import os
import collections
import gzip
import pickle
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
_NORMAL = 1
_MINIMAL = 0


class BLAST(object):

    """
    Cell BLAST algorithm

    Parameters
    ----------
    models : list
        List of ``DIRECTi`` models, must be specified.
    ref : Cell_BLAST.data.ExprDataSet
        Reference dataset, must be specified.
    keep_exprs : bool
        Whether to store reference dataset expression, by default False.
        Not that to predict expression level or do aligning BLAST later,
        this must be set to True.
    n_posterior : int
        How many samples from the posterior distribution to use for
        estimating posterior distance, by default 50. If set to 0, only
        Euclidean distance will be used downstream.
    n_jobs : int
        Number of parallel jobs to run when building the BLAST index. If not
        specified, ``Cell_BLAST.config.N_JOBS`` will be used, which defaults
        to 1. Note that each (tensorflow) job could be distributed on
        multiple CPUs for a single "job".
    random_seed : int
        Random seed for posterior sampling. If not specified,
        ``Cell_BLAST.config.RANDOM_SEED`` will be used, which defaults to None.

    Examples
    --------

    A typical BLAST pipeline is described below.

    Assuming we have a list of ``DIRECTi`` models already fitted on normalized
    reference data, we can then construct a BLAST object. Data structure
    required to efficiently perform cell BLAST is built behind the scene.
    Also, we use the ``build_empirical`` method to build an empirical
    distribution of posterior distance for computation of empirical p-values
    that will be used later.

    >>> blast = BLAST(models, reference).build_empirical()

    Then we efficiently query the database and obtain initial hits (query
    dataset should be normalized in the same way as the reference dataset):

    >>> hits = blast.query(query)

    Then we filter the initial hits through the use of more accurate metrics
    (e.g. empirical p-value based on posterior distance), and pooling together
    information across multiple models.

    >>> hits = hits.reconcile_models().filter(by="pval", cutoff=0.05)

    We can print the ``hits`` object to see annotations of reference hits
    as well as similarity and significance of each hit.

    >>> print(hits)

    Finally, we use the ``annotate`` method to obtain query cell annotations
    based on reference annotations, e.g. "cell_ontology_class" in this case.

    >>> annotation = hits.annotate("cell_ontology_class")

    See the BLAST ipython notebook (:ref:`vignettes`) for live examples.
    """

    def __init__(self, models=None, ref=None, keep_exprs=False,
                 n_posterior=50, n_jobs=config._USE_GLOBAL,
                 random_seed=config._USE_GLOBAL):

        assert (models is None) == (ref is None)
        if models is None and ref is None:
            return  # Only to be used in BLAST.load(...)

        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed
        self.models = models
        self.n_posterior = n_posterior

        message.info("Projecting to latent space...")
        clean_latent, noisy_latent = zip(*joblib.Parallel(
            n_jobs=min(n_jobs, len(self.models)), backend="threading"
        )(joblib.delayed(_inference)(
            model, ref, self.n_posterior, random_seed
        ) for model in self.models))
        # clean_latent is a list of n_cells * latent_dim
        # noisy_latent is a list of n_cells * posterior_samples * latent_dim

        message.info("Fitting nearest neighbor trees...")
        self.nearest_neighbors = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_fit_nearest_neighbors)(
                _clean_latent
            ) for _clean_latent in clean_latent
        )

        self.clean_latent = np.stack(clean_latent, axis=0)  # n_models * n_cells * latent_dim
        self.noisy_latent = np.stack(noisy_latent, axis=0) \
            if self.n_posterior else None  # n_models * n_cells * n_posterior_samples * latent_dim

        self.ref = ref.copy() if keep_exprs else ref[:, []]
        self.ref.uns = {}  # uns won't be used
        self.empirical = None
        self._mode = _NORMAL

    def build_empirical(self, background=None, n_empirical=10000,
                        n_jobs=config._USE_GLOBAL, random_seed=config._USE_GLOBAL):
        """
        Build empirical distribution of posterior distance

        Parameters
        ----------
        background : Cell_BLAST.data.ExprDataSet
            dataset to use when building this empirical distribution,
            by default None, meaning that the reference dataset itself will be
            used as background.
        n_empirical : int
            Number of random cell pairs to use when estimating empirical
            distribution of posterior distance, by default 10000.
            Setting ``n_empirical=0`` skips building empirical distribution, and
            empirical p-values will not be available.
        n_jobs : int
            Number of parallel jobs to run. If not specified,
            ``Cell_BLAST.config.N_JOBS`` will be used, which defaults to 1.
            Note that each (tensorflow) job could be distributed on multiple
            CPUs for a single "job".
        random_seed : int
            Random seed for posterior sampling. If not specified,
            ``Cell_BLAST.config.RANDOM_SEED`` will be used,
            which defaults to None.

        Returns
        -------
        blast : Cell_BLAST.blast.BLAST
            A BLAST object with empirical posterior distance distribution built.
        """
        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed
        if background is not None:
            message.info("Projecting to latent space...")
            clean_latent, noisy_latent = zip(*joblib.Parallel(
                n_jobs=min(n_jobs, len(self.models)), backend="threading"
            )(joblib.delayed(_inference)(
                model, background, self.n_posterior, random_seed
            ) for model in self.models))
            clean_latent = np.stack(clean_latent, axis=0)
            noisy_latent = np.stack(noisy_latent, axis=0) \
                if self.n_posterior else None
        else:
            clean_latent, noisy_latent = self.clean_latent, self.noisy_latent

        # Build an empirical distribution of posterior distance
        random_state = np.random.RandomState(random_seed)
        random_pairs = np.stack([
            random_state.randint(  # Reference index
                low=0, high=self.clean_latent.shape[1], size=n_empirical),
            random_state.randint(  # Background index
                low=0, high=clean_latent.shape[1], size=n_empirical),
        ], axis=1)
        if self.n_posterior:
            message.info("Computing posterior distribution distances...")
            self.empirical = joblib.Parallel(
                n_jobs=n_jobs, backend="threading"
            )(joblib.delayed(_hit_posterior_distance_across_models)(
                self.clean_latent[:, [pair[0]], ...], clean_latent[:, pair[1], ...],
                self.noisy_latent[:, [pair[0]], ...], noisy_latent[:, pair[1], ...]
            ) for pair in random_pairs)  # list of n_models * 1
        else:
            message.info("Computing Euclidean distances...")
            self.empirical = joblib.Parallel(
                n_jobs=n_jobs, backend="threading"
            )(joblib.delayed(_hit_euclidean_distance_across_models)(
                self.clean_latent[:, [pair[0]], ...],
                clean_latent[:, pair[1], ...]
            ) for pair in random_pairs)  # list of n_models * 1
        self.empirical = np.concatenate(self.empirical, axis=1)  # n_models * n_empirical
        return self

    def __len__(self):
        return len(self.models)

    def __getitem__(self, slice):
        blast = BLAST()
        blast.models = self.models[slice]
        blast.clean_latent = self.clean_latent[slice]
        blast.noisy_latent = self.noisy_latent[slice]
        blast.nearest_neighbors = self.nearest_neighbors[slice]
        blast.empirical = self.empirical[slice] \
            if self.empirical is not None else None
        blast.ref = self.ref
        blast.n_posterior = self.n_posterior
        blast._mode = self._mode
        return blast

    def save(self, path):
        """
        Save models and index to a path.

        Parameters
        ----------
        path : str
            Specifies where to save the index.
        """
        if self._mode == _MINIMAL:  # pragma: no cover
            raise Exception("Save not available in MINIMAL mode!")
        if not os.path.exists(path):
            os.makedirs(path)
        with gzip.open(os.path.join(
            path, "index.pkz"
        ), "wb", compresslevel=7) as f:
            pickle.dump(dict(
                nearest_neighbors=self.nearest_neighbors,
                empirical=self.empirical
            ), f)
        if self.noisy_latent is not None:
            data.write_hybrid_path(self.noisy_latent, os.path.join(
                path, "index.h5//noisy_latent"))
        if self.ref is not None:
            self.ref.write_dataset(os.path.join(path, "ref.h5"))
        for i in range(len(self.models)):
            self.models[i].save(os.path.join(path, "model_%d" % i))

    @classmethod
    def load(cls, path, skip_exprs=False, mode="normal",
             model_slice=None, verbose=1):
        """
        Load models and index from file.

        Parameters
        ----------
        path : str
            Specifies a path containing models and index.
        skip_exprs : bool
            Whether to skip loading expression values, by default False
        mode : {"normal", "minimal"}
            If mode is set to "minimal", model loading will accelerate by only
            loading the encoders, but aligning BLAST would not be available.
        model_slice : slice, None
            Use only a subset of models, by default None, meaning all
            available models are used.
        verbose : int
            Controls model loading verbosity, by default 1.
            Check ``Cell_BLAST.model.Model`` for details.

        Returns
        -------
        blast : Cell_BLAST.blast.BLAST
            Loaded BLAST object.
        """
        assert mode in ("normal", "minimal")
        if model_slice is None:
            model_slice = slice(None)
        blast = cls()
        with gzip.open(os.path.join(path, "index.pkz"), "rb") as f:
            index = pickle.load(f)
        blast.nearest_neighbors = index["nearest_neighbors"][model_slice]
        blast.clean_latent = np.stack([
            nearest_neighbor._fit_X
            for nearest_neighbor in blast.nearest_neighbors
        ], axis=0)
        blast.empirical = index["empirical"][model_slice] \
            if index["empirical"] is not None else None

        if "ref" in index:  # pragma: no cover
            # Compatibility with old index
            message.info("Compatibility mode: using ref data in \"index.pkz\".")
            blast.ref = index["ref"]
        else:
            blast.ref = data.ExprDataSet.read_dataset(os.path.join(
                path, "ref.h5"), skip_exprs=skip_exprs)

        if "noisy_latent" in index:  # pragma: no cover
            message.info("Compatibility mode: using noisy latent in \"index.pkz\".")
            blast.noisy_latent = index["noisy_latent"]
        elif os.path.exists(os.path.join(path, "index.h5")):
            # with h5py.File(os.path.join(path, "index.h5"), "r") as f:
            #     blast.noisy_latent = f["noisy_latent"][...]
            blast.noisy_latent = data.read_hybrid_path(os.path.join(
                path, "index.h5//noisy_latent"))
        else:
            blast.noisy_latent = None
        blast.noisy_latent = blast.noisy_latent[model_slice] \
            if blast.noisy_latent is not None else None
        blast.n_posterior = blast.noisy_latent.shape[2] if \
            blast.noisy_latent is not None else 0

        blast.models = []
        for i in np.arange(len(index["nearest_neighbors"]))[model_slice]:
            blast.models.append(directi.DIRECTi.load(
                os.path.join(path, "model_%d" % i),
                _mode=mode == "normal", verbose=verbose
            ))
        blast._mode = _MINIMAL if mode == "minimal" else _NORMAL
        return blast

    def query(self, query, n_neighbors=5, n_jobs=config._USE_GLOBAL,
              random_seed=config._USE_GLOBAL):
        """
        BLAST query

        Parameters
        ----------
        query : Cell_BLAST.data.ExprDataSet, array_like
            Query transcriptomes in the form of either an ``ExprDataSet`` or
            a 2d-array of shape :math:`cell \\times gene`. Note that proper
            normalization should be performed beforehand.
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
        clean_latent, noisy_latent = zip(*joblib.Parallel(
            n_jobs=min(n_jobs, len(self.models)), backend="threading"
        )(joblib.delayed(_inference)(
            model, query, self.n_posterior, random_seed
        ) for model in self.models))
        clean_latent = np.stack(clean_latent, axis=0)  # n_models * n_cells * latent_dim
        noisy_latent = np.stack(noisy_latent, axis=0)  # n_models * n_cells * n_posterior_samples * latent_dim

        message.info("Doing nearest neighbor search...")
        nni = joblib.Parallel(
            n_jobs=min(n_jobs, len(self.models)), backend="threading"
        )(
            joblib.delayed(_nearest_neighbor_search)(
                self.nearest_neighbors[k], clean_latent[k, ...], n_neighbors
            ) for k in range(len(self.models))
        )
        nni = np.stack(nni, axis=2)  # n_cells * n_neighbors * n_models

        message.info("Merging hits across models...")
        hits = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
            joblib.delayed(_nearest_neighbor_merge)(_nni) for _nni in nni
        )  # list of n_hits

        if self.n_posterior:
            message.info("Computing posterior distribution distances...")
            dist = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_hit_posterior_distance_across_models)(
                    self.clean_latent[:, hits[i], ...], clean_latent[:, i, ...],
                    self.noisy_latent[:, hits[i], ...], noisy_latent[:, i, ...]
                ) for i in range(len(hits))
            )  # list of n_models * n_hits
        else:
            message.info("Computing Euclidean distances...")
            dist = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_hit_euclidean_distance_across_models)(
                    self.clean_latent[:, hits[i], ...], clean_latent[:, i, ...]
                ) for i in range(len(hits))
            )  # list of n_models * n_hits

        if self.empirical is not None:
            message.info("Computing empirical p-values...")
            pval = joblib.Parallel(
                n_jobs=n_jobs, backend="threading"
            )(
                joblib.delayed(_empirical_pvalue)(
                    _dist, self.empirical
                ) for _dist in dist
            )  # list of n_models * n_hits
        else:
            pval = None

        return Hits(
            self.ref, list(hits), list(dist),
            list(pval) if pval is not None else None,
            query.obs_names if isinstance(query, data.ExprDataSet) else None
        )

    def align(self, query, n_jobs=config._USE_GLOBAL,
              random_seed=config._USE_GLOBAL, path=".", **kwargs):
        """
        Align internal DIRECTi models with queries

        Parameters
        ----------
        query : Cell_BLAST.data.ExprDataSet, dict
            Query dataset or a dict of query datasets, which will be aligned
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
            Specifies a path to store temporary files, by default ".".
        **kwargs
            Additional keyword parameters will be passed to
            ``Cell_BLAST.directi.align_DIRECTi``.

        Returns
        -------
        blast : Cell_BLAST.blast.BLAST
            A new BLAST object with aligned internal models.
        """
        if self._mode == _MINIMAL:  # pragma: no cover
            raise Exception("Align not available in MINIMAL mode!")
        n_jobs = config.N_JOBS if n_jobs == config._USE_GLOBAL else n_jobs
        random_seed = config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed

        aligned_models = joblib.Parallel(
            n_jobs=n_jobs, backend="threading"
        )(
            joblib.delayed(directi.align_DIRECTi)(
                self.models[i], self.ref, query, random_seed=random_seed,
                path=os.path.join(path, "aligned_model_%d" % i), **kwargs
            ) for i in range(len(self.models))
        )
        return BLAST(models=aligned_models, ref=self.ref, keep_exprs=True,
                     n_posterior=self.n_posterior, n_jobs=n_jobs,
                     random_seed=random_seed)


class Hits(object):

    """
    BLAST hits

    Parameters
    ----------
    ref : Cell_BLAST.data.ExprDataSet
        Reference dataset.
    hits : list
        Reference hit indices.
        Each list element corresponds to a cell in the query,
        and contains indices of its reference hits.
    dist : list
        Reference hit distances. Each list element corresponds to a cell
        in the query, and contains distances (euclidean or posterior) to its
        reference hits in each model respectively (thus in a shape of
        :math:`n\\_models \\times n\\_hits`).
    pval : list, None
        Reference hit empirical p-values, by default None.
        Each list element corresponds to a cell in the query,
        and contains empirical p-values of its reference hits in each
        model respectively (thus in a shape of
        :math:`n\\_models \\times n\\_hits`).
    names : array_like, None
        Query cell names, by default None.
    """

    FILTER_BY_DIST = 0
    FILTER_BY_PVAL = 1

    def __init__(self, ref, hits, dist, pval=None,
                 names=None, _reconciled_flag=False):
        self.ref = ref
        self.hits = hits
        self.dist = dist
        self.pval = pval
        self.names = names if names is not None else np.arange(ref.shape[0])
        self._reconciled_flag = _reconciled_flag

    # def __iter__(self):
    #     hits, dist, names = self.hits, self.dist, self.names
    #     pval = self.pval if self.pval is not None else [None] * len(self.hits)
    #     for _hits, _dist, _pval, _name in zip(hits, dist, pval, names):
    #         yield Hits(self.ref, [_hits], [_dist], [_pval],
    #                    [_name], self._reconciled_flag)

    def to_data_frames(self):
        """
        Construct a hit data frame for each cell.
        Note that only reconciled ``Hits`` objects are supported.

        Returns
        -------
        data_frame_dict : dict
            Each element is hit data frame for a cell
        """
        assert self._reconciled_flag
        df_dict = collections.OrderedDict()
        for i, name in enumerate(self.names):
            df_dict[name] = self.ref.obs.iloc[self.hits[i], :]
            df_dict[name]["hits"] = self.hits[i]
            df_dict[name]["dist"] = self.dist[i].ravel()
            if self.pval is not None:
                df_dict[name]["pval"] = self.pval[i].ravel()
        return df_dict

    def reconcile_models(self, dist_method="mean", pval_method="gmean"):
        """
        Merge model specific distances and empirical p-values.

        Parameters
        ----------
        dist_method : {"mean", "gmean", "min", "max"}
            Specifies how to reconcile distances across difference models,
            by default "mean".
        pval_method : {"mean", "gmean", "min", "max"}
            Specifies how to reconcile empirical p-values across
            different models, by default "gmean".

        Returns
        -------
        reconciled_hits : Cell_BLAST.blast.Hits
            Hit object containing reconciled
        """
        dist_method = self._get_reconcile_method(dist_method)
        dist = [np.expand_dims(dist_method(
            item, axis=0
        ), axis=0) for item in self.dist]
        if self.pval is not None:
            pval_method = self._get_reconcile_method(pval_method)
            pval = [np.expand_dims(pval_method(
                item, axis=0
            ), axis=0) for item in self.pval]
        else:
            pval = None
        return Hits(self.ref, self.hits, dist, pval, self.names, True)

    def filter(self, by="pval", cutoff=0.05, model_tolerance=0, n_jobs=1):
        """
        Filter hits by posterior distance or p-value

        Parameters
        ----------
        by : {"dist", "pval"}
            Specify a metric based on which to filter hits, by default "pval".
        cutoff : float
            Metric cutoff when filtering hits, by default 0.05.
        model_tolerance : int
            Maximal number of models in which cutoff is not reached, for a
            hit to be retained, by default 0.
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
        result = [_ for _ in zip(*joblib.Parallel(
            n_jobs=n_jobs, backend="threading"
        )(
            joblib.delayed(_filter_hits)(
                self.hits[i], self.dist[i],
                self.pval[i] if self.pval is not None
                    else -np.ones_like(self.dist[i]),
                by, cutoff, model_tolerance
            ) for i in range(len(self.hits))
        ))]
        if self.pval is None:
            return Hits(self.ref, list(result[0]), list(result[1]), None,
                        self.names, self._reconciled_flag)
        return Hits(self.ref, list(result[0]), list(result[1]), list(result[2]),
                    self.names, self._reconciled_flag)

    def annotate(self, field, min_hits=2, majority_threshold=0.5):
        """
        Annotate query with information from the reference.
        Fields in the meta data or expression values can all be specified for
        annotation. Note that predicted expression values are cell normalized
        (to 1e4) and in log scale.

        Parameters
        ----------
        field : str or array_like
            Specifies meta columns or variable names to use for
            annotating query. Note that numerical variables and categorical
            variables cannot be mixed and only one categorical variable is
            supported at a time.
        min_hits : int
            Minimal number of hits required for annotating a cell,
            otherwise the query will be rejected, by default 2.
        majority_threshold : float
            Minimal majority fraction (not inclusive) required for prediction,
            by default 0.5. Only effective when predicting categorical
            variables. If threshold is not met, annotation will be "ambiguous".

        Returns
        -------
        prediction : pandas.DataFrame
            Each row contains the inferred annotation for a query cell.
        """
        if isinstance(field, str):
            field = [field]
        ref = self.ref.get_meta_or_var(
            field, normalize_var=True, log_var=True
        ).values
        prediction = []
        if ref.dtype.type in (np.str_, np.string_, np.object_):
            if ref.ndim == 2:
                if ref.shape[1] > 1:  # pragma: no cover
                    raise ValueError("Only one categorical variable "
                                     "can be predicted at a time!")
                ref = ref.ravel()
            for _hits in self.hits:
                if len(_hits) < min_hits:
                    prediction.append("rejected")
                    continue
                hits = ref[_hits]
                label, count = np.unique(hits, return_counts=True)
                best_idx = count.argmax()
                if count[best_idx] / len(hits) <= majority_threshold:
                    prediction.append("ambiguous")
                    continue
                prediction.append(label[best_idx])
            prediction = np.array(prediction).reshape((-1, 1))
        elif np.issubdtype(ref.dtype.type, np.number):
            for _hits in self.hits:
                if len(_hits) < min_hits:
                    prediction.append(np.empty(len(field)) * np.nan)
                    continue
                prediction.append(np.array(ref[_hits].mean(axis=0)))
                # np.array call is for 1-d mean that produces 0-d values
            prediction = np.stack(prediction, axis=0)
        else:  # pragma: no cover
            raise ValueError("Unsupported annotation columns!")
        return pd.DataFrame(prediction, index=self.names, columns=field)

    def __getitem__(self, slice):
        return Hits(
            self.ref, self.hits[slice], self.dist[slice],
            self.pval[slice] if self.pval is not None else None,
            self.names[slice], self._reconciled_flag
        )

    def __str__(self):
        string = []
        for idx, hits in enumerate(self.hits):
            string.append(">>> %s:" % self.names[idx])
            df = self.ref.obs.iloc[hits, :]
            dist = self.dist[idx]
            if self.pval is None:
                pval = np.ones_like(dist) * np.nan
            else:
                pval = self.pval[idx]
            if self._reconciled_flag:
                df.loc[:, "distance"] = dist.ravel()
                df.loc[:, "p-value"] = pval.ravel()
                if not np.any(np.isnan(df["p-value"])):
                    df = df.sort_values("p-value")
                else:
                    df = df.sort_values("distance")
            else:
                for model, (_dist, _pval) in enumerate(zip(dist, pval)):
                    df.loc[:, "distance (model_%d)" % model] = _dist
                    df.loc[:, "p-value (model_%d)" % model] = _pval
            string.append(str(df))
            string.append("\n")
        return "\n".join(string)

    def __len__(self):
        return len(self.names)

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


def _inference(model, dataset, n_posterior, random_seed=config._USE_GLOBAL):
    if not isinstance(dataset, data.ExprDataSet):
        assert dataset.shape[1] == len(model.genes)
    clean_latent = model.inference(
        dataset, progress_bar=False
    ).astype(np.float32)
    noisy_latent = model.inference(
        dataset, noisy=n_posterior, random_seed=random_seed, progress_bar=False
    ).astype(np.float32) if n_posterior else None
    return clean_latent, noisy_latent


def _fit_nearest_neighbors(x):
    return sklearn.neighbors.NearestNeighbors().fit(x)


def _nearest_neighbor_search(nn, query, n_neighbors):
    return nn.kneighbors(query, n_neighbors=n_neighbors)[1]


@numba.jit(nopython=True, nogil=True, cache=True)
def _nearest_neighbor_merge(x):  # pragma: no cover
    return np.unique(x)


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
def _hit_posterior_distance_across_models(
    ref_clean, query_clean, ref_noisy, query_noisy, normalize=True
):  # pragma: no cover
    """
    ref_clean : n_models * n_hits * latent_dim
    query_clean : n_models * latent_dim
    ref_noisy : n_models * n_hits * n_posterior_samples * latent_dim
    query_noisy : n_models * n_posterior_samples * latent_dim
    """
    dist = np.empty(ref_clean.shape[:-1])  # n_models * n_hits
    for i in range(dist.shape[0]):  # model index
        for j in range(dist.shape[1]):  # hit index
            x = ref_clean[i, j, ...]  # latent_dim
            y = query_clean[i, ...]  # latent_dim
            noisy_x = ref_noisy[i, j, ...]  # n_posterior_samples * latent_dim
            noisy_y = query_noisy[i, ...]  # n_posterior_samples * latent_dim

            projection = (x - y).reshape((-1, 1))  # latent_dim * 1
            if np.all(projection == 0):
                projection[...] = 1  # any projection is equivalent
            projection /= np.linalg.norm(projection)
            scalar_noisy_x = np.dot(noisy_x, projection).ravel()  # n_posterior_samples
            scalar_noisy_y = np.dot(noisy_y, projection).ravel()  # n_posterior_samples
            # TODO: there is a numba warning saying that one of np.dot arguments
            # is not contiguous, but they should be both contiguous here.
            # Haven't figured out why...
            noisy_xy = np.concatenate((
                scalar_noisy_x, scalar_noisy_y
            ), axis=0)
            if normalize:
                noisy_xy1 = (noisy_xy - np.mean(scalar_noisy_x)) / np.std(scalar_noisy_x)
                noisy_xy2 = (noisy_xy - np.mean(scalar_noisy_y)) / np.std(scalar_noisy_y)
                dist[i, j] = scipy.stats.wasserstein_distance(
                    noisy_xy1[:len(scalar_noisy_x)],
                    noisy_xy1[-len(scalar_noisy_y):]
                ) + scipy.stats.wasserstein_distance(
                    noisy_xy2[:len(scalar_noisy_x)],
                    noisy_xy2[-len(scalar_noisy_y):]
                )
                dist[i, j] /= 2
            else:
                dist[i, j] = scipy.stats.wasserstein_distance(
                    noisy_xy[:len(scalar_noisy_x)],
                    noisy_xy[-len(scalar_noisy_y):]
                )
    return dist


@numba.jit(nopython=True, nogil=True, cache=True)
def _hit_euclidean_distance_across_models(ref_clean, query_clean):  # pragma: no cover
    """
    ref_clean : n_models * n_hits * latent_dim
    query_clean : n_models * latent_dim
    """
    return np.sqrt(np.square(
        ref_clean - np.expand_dims(query_clean, axis=1)
    ).sum(axis=2))


@numba.jit(nopython=True, nogil=True, cache=True)
def _empirical_pvalue(dist, emp):  # pragma: no cover
    """
    dist : n_models * n_hits
    emp : n_models * n_empirical
    """
    return (
        np.expand_dims(emp, axis=1) <=  # n_models * 1 * n_empirical
        np.expand_dims(dist, axis=2)  # n_models * n_hits * 1
    ).sum(axis=2) / emp.shape[1]  # n_models * n_hits


@numba.jit(nopython=True, nogil=True, cache=True)
def _filter_hits(hits, dist, pval, by, cutoff, model_tolerance):  # pragma: no cover
    """
    hits : n_hits
    dist : n_models * n_hits
    pval : n_models * n_hits
    """
    if by == 0:  # Hits.FILTER_BY_DIST
        pair_mask = dist <= cutoff
        hit_mask = (dist.shape[0] - pair_mask.sum(axis=0)) <= model_tolerance
    else:  # Hits.FILTER_BY_PVAL
        pair_mask = pval <= cutoff
        hit_mask = (pval.shape[0] - pair_mask.sum(axis=0)) <= model_tolerance
    if np.all(pval == -1):
        return hits[hit_mask], dist[:, hit_mask], pval
    return hits[hit_mask], dist[:, hit_mask], pval[:, hit_mask]


def sankey(query, ref, title="Sankey", width=500, height=500, tint_cutoff=1,
           suppress_plot=False):  # pragma: no cover
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
            size=10
        )
    )

    fig = dict(data=[sankey_data], layout=sankey_layout)
    if not suppress_plot:
        import plotly.offline
        plotly.offline.init_notebook_mode()
        plotly.offline.iplot(fig, validate=False)
    return fig
