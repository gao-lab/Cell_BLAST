"""
DIRECTi, an deep learning model for semi-supervised parametric dimension
reduction and systematical bias removal, extended from scVI.
"""


import os
import json
import numpy as np
import pandas as pd
import scipy.sparse
import tensorflow as tf
from . import model
from . import message
from . import utils
from . import latent
from . import rmbatch
from . import prob
from . import data
from . import config


_TRAIN = 1
_TEST = 0


class DIRECTi(model.Model):
    """
    DIRECTi model.

    Parameters
    ----------
    genes : pandas.Series, numpy.ndarray (1d), list
        Genes to use in the model.
    latent_module : Cell_BLAST.latent.Latent
        Module for latent variable / encoder.
    prob_module : Cell_BLAST.prob.ProbModel
        Module for data probabilistic model / decoder.
    rmbatch_modules : list
        List of modules for systematical bias / batch effect removal,
        by default None.
    denoising : bool
        Whether to add noise to input during training, by default True.
    decoder_feed_batch : str, bool
        How to feed batch information to the decoder, by default "nonlinear".
        Available options are listed below:
        "nonlinear": concatenate with input layer and go through all nonlinear
        transformations in the decoder;
        "linear": concatenate with last hidden layer and only go through
        the last linear transformation (potentially one last non-linear
        transformation);
        "both": concatenate with both input layer and last hidden layer;
        False: do not feed batch information to decoder.
    path : str
        Specifies a path where model configuration, checkpoints,
        as well as the final model will be saved, by default "."
    random_seed : int
        Random seed. If not specified, ``Cell_BLAST.config.RANDOM_SEED``
        will be used, which defaults to None.

    Attributes
    ----------
    genes : list
        List of gene names the model is defined and fitted on

    Examples
    --------

    :ref:`fit_DIRECTi` offers an easy to use wrapper of this ``DIRECTi`` model,
    which should satisfy most needs. We suggest using that wrapper first.
    However, if you do wish to directly use the ``DIRECTi`` class, here's a
    brief instruction:

    First you need to select proper modules. ``DIRECTi`` is made up of three
    kinds of modules, a latent (encoder) module, a probabilistic model (decoder)
    module and optional batch effect removal modules:

    >>> latent_module = Cell_BLAST.latent.Gau(latent_dim=10)
    >>> prob_module = Cell_BLAST.prob.ZINB()
    >>> rmbatch_modules = [Cell_BLAST.rmbatch.Adversarial(
    ...     batch_dim=2, name="rmbatch"
    ... )]

    We also need a list of gene names which defines the gene set on which
    the model is fitted. It can also be accessed later to help ensure correct
    gene set of the input data.
    Then we can assemble them into a complete DIRECTi model and compile it:

    >>> model = Cell_BLAST.directi.DIRECTi(
    ...     genes, latent_module, prob_module, rmbatch_modules
    ... ).compile(optimizer="RMSPropOptimizer", lr=1e-3)

    To fit the model, we need a ``Cell_BLAST.utils.DataDict`` object,
    which can be thought of as a dict of array-like objects.

    In the ``DataDict``, we require an "exprs" slot that stores the
    :math:`cell \\times gene` expression matrix.
    If the latent module ``SemiSupervisedCatGau`` is used, we additionally
    require a slot containing one-hot encoded supervision labels. Unsupervised
    cells should have all zeros in the corresponding rows. Name of the slot
    should be the same as name of the semi-supervision latent module.
    If batch effect removal modules are used, we additionally require slots
    containing one-hot encoded batch identity. Name of the slots should be
    the same as name of batch removal modules.

    Here's how to construct a data dict from a ``Cell_BLAST.data.ExprDataSet``
    object:

    >>> data_dict = Cell_BLAST.utils.DataDict(
    ...     exprs=data_obj[:, model.genes].exprs,
    ...     rmbatch=Cell_BLAST.utils.encode_onehot(data_obj.obs["batch"])
    ... )

    Now we are ready to fit the model:

    >>> model.fit(data_dict)

    At this point we have obtained a fitted DIRECTi model object, which is also
    the return value if the ``fit_DIRECTi`` function.
    We can use this model to project transriptomes into the low dimensional
    latent space by using the ``inference`` method. You may pass it to
    ``latent`` slot in the original data object, which facilitates subsequent
    visualization of the latent space:

    >>> data_obj.latent = model.inference(data_obj)
    >>> data_obj.visualize_latent("cell_type")

    """
    _TRAIN = 1
    _TEST = 0

    def __init__(
        self, genes, latent_module, prob_module, rmbatch_modules=None,
        denoising=True, decoder_feed_batch="nonlinear",
        path=".", random_seed=config._USE_GLOBAL, _mode=_TRAIN
    ):
        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed
        if isinstance(genes, pd.Series):
            genes = genes.values
        if isinstance(genes, np.ndarray):
            genes = genes.tolist()
        assert isinstance(genes, list)
        self.genes = genes
        self.x_dim = len(genes)
        self._mode = _mode
        super(DIRECTi, self).__init__(
            latent_module=latent_module,
            prob_module=prob_module,
            rmbatch_modules=rmbatch_modules,
            denoising=denoising,
            decoder_feed_batch=decoder_feed_batch,
            random_seed=random_seed,
            path=path
        )

    def _init_graph(
        self, latent_module, prob_module, rmbatch_modules=None,
        denoising=True, decoder_feed_batch="nonlinear"
    ):
        super(DIRECTi, self)._init_graph()
        self.denoising = denoising
        self.decoder_feed_batch = decoder_feed_batch
        self.latent_module = latent_module
        self.prob_module = prob_module
        self.rmbatch_modules = [] if rmbatch_modules is None else rmbatch_modules
        self.loss, self.early_stop_loss = [], []

        with tf.name_scope("placeholder/"):
            self.x = tf.placeholder(
                dtype=tf.float32, shape=(None, self.x_dim), name="x")
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag")

        # Preprocessing
        self.noisy_x = prob_module._add_noise(self.x) \
            if denoising else self.x
        preprocessed_x = prob_module._preprocess(self.noisy_x)

        # Encoder
        self.latent = self.latent_module._build_latent(
            preprocessed_x, self.training_flag, scope="encoder")
        if self._mode == _TEST:
            return
        self.loss.append(self.latent_module._build_regularizer(
            self.training_flag, self.epoch))

        # Remove batch effect
        for rmbatch_module in self.rmbatch_modules:
            self.loss.append(rmbatch_module._build_regularizer(
                self.latent, self.training_flag, self.epoch))

        # Decoder
        feed_batch = [
            rmbatch_module.batch for rmbatch_module in self.rmbatch_modules
        ] if decoder_feed_batch in (
            "nonlinear", "linear", "both"
        ) and self.rmbatch_modules else None
        latent = [self.latent] + feed_batch if decoder_feed_batch in (
            "nonlinear", "both"
        ) and self.rmbatch_modules else [self.latent]
        tail_concat = feed_batch if decoder_feed_batch in (
            "linear", "both"
        ) and self.rmbatch_modules else None
        recon_loss = self.prob_module._loss(
            self.x, latent, self.training_flag,
            tail_concat=tail_concat, scope="decoder"
        )

        self.loss.append(recon_loss)
        self.early_stop_loss.append(recon_loss)

        self.loss = tf.add_n([
            item for item in
            self.loss + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if item != 0
        ], name="total_loss")
        self.early_stop_loss = tf.add_n([
            item for item in self.early_stop_loss if item != 0
        ], name="early_stop_loss")
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.early_stop_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.loss)

    def _compile(self, optimizer="RMSPropOptimizer", lr=1e-3):
        if self.latent_module:
            self.latent_module._compile(optimizer, lr)
        if self.prob_module:
            self.prob_module._compile(optimizer, lr)
        for rmbatch_module in self.rmbatch_modules:
            rmbatch_module._compile(optimizer, lr)
        with tf.variable_scope("optimize/main"):
            control_dependencies = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(control_dependencies):
                self.step = tf.train.__dict__[optimizer](lr).minimize(
                    self.loss, var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "encoder"
                    ) + tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "decoder"
                    )
                )

    def _fit_epoch(self, data_dict, batch_size=128, progress_bar=True):
        loss_dict = {
            item: 0.0 for item in tf.get_collection(tf.GraphKeys.LOSSES)
        }

        # Training
        @utils.minibatch(batch_size, desc="training",
                         use_last=False, progress_bar=progress_bar)
        def _train(data_dict):
            nonlocal loss_dict
            feed_dict = {
                self.x: utils.densify(data_dict["exprs"]),
                **(self.latent_module._build_feed_dict(data_dict)),
                **(self.prob_module._build_feed_dict(data_dict)),
                self.training_flag: True
            }
            for rmbatch_module in self.rmbatch_modules:
                feed_dict.update(rmbatch_module._build_feed_dict(data_dict))
            run_result = self.sess.run(
                [self.step] + list(loss_dict.keys()), feed_dict=feed_dict)
            run_result.pop(0)
            for item in loss_dict:
                loss_dict[item] += run_result.pop(0) * data_dict.size

        _train(data_dict)
        for item in loss_dict:
            loss_dict[item] /= data_dict.size
        self.epoch_report += "train=%.3f, " % loss_dict[self.early_stop_loss]

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="%s (train)" % item.name,
                             simple_value=loss_dict[item])
            for item in loss_dict
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(self, data_dict, batch_size=128, progress_bar=True):
        loss_dict = {
            item: 0.0 for item in tf.get_collection(tf.GraphKeys.LOSSES)
        }

        # Validation
        @utils.minibatch(batch_size, desc="validation",
                         use_last=True, progress_bar=progress_bar)
        def _validate(data_dict):
            nonlocal loss_dict
            feed_dict = {
                self.x: utils.densify(data_dict["exprs"]),
                **(self.latent_module._build_feed_dict(data_dict)),
                **(self.prob_module._build_feed_dict(data_dict)),
                self.training_flag: False
            }
            for rmbatch_module in self.rmbatch_modules:
                feed_dict.update(rmbatch_module._build_feed_dict(data_dict))
            run_result = self.sess.run(
                list(loss_dict.keys()),
                feed_dict=feed_dict)
            for item in loss_dict:
                loss_dict[item] += run_result.pop(0) * data_dict.size

        _validate(data_dict)
        for item in loss_dict:
            loss_dict[item] /= data_dict.size
        self.epoch_report += "val=%.3f, " % loss_dict[self.early_stop_loss]

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="%s (val)" % item.name,
                             simple_value=loss_dict[item])
            for item in loss_dict
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss_dict[self.early_stop_loss]

    def fit(self, data_dict, batch_size=128, val_split=0.1, epoch=100,
            patience=np.inf, on_epoch_end=None, progress_bar=False):
        """
        Fit the model.

        Parameters
        ----------
        data_dict : Cell_BLAST.utils.DataDict
            Training data.
        batch_size : int
            Size of minibatch used in training, by default 128.
        val_split : float
            Fraction of data to use for validation, by default 0.1.
        epoch : int
            Maximal training epochs, by default 100.
        patience : int
            Early stop patience, by default numpy.inf, meaning no early stop.
            Model training stops when best validation loss does not decrease
            for a consecutive ``patience`` epochs.
        on_epoch_end : list
            List of functions to be executed at the end of each epoch,
            by default None.
        progress_bar : bool
            Whether to print progress bar for each epoch during training,
            by default True.

        Returns
        -------
        model : DIRECTi
            The fitted model
        """
        if on_epoch_end is None:
            on_epoch_end = []
        on_epoch_end += \
            self.latent_module.on_epoch_end + \
            self.prob_module.on_epoch_end
        for rmbatch_module in self.rmbatch_modules:
            on_epoch_end += rmbatch_module.on_epoch_end
        return super(DIRECTi, self).fit(
            data_dict, batch_size=batch_size, val_split=val_split, epoch=epoch,
            patience=patience, on_epoch_end=on_epoch_end,
            progress_bar=progress_bar
        )

    @utils.with_self_graph
    def _fetch(self, tensor, x=None, batch_size=4096,
               noisy=False, progress_bar=False, random_seed=config._USE_GLOBAL):
        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed

        if x is None:
            return self.sess.run(tensor)

        tensor_shape = tuple(
            item for item in tensor.get_shape().as_list() if item is not None)
        result = np.empty((x.shape[0],) + tuple(tensor_shape))
        random_state = np.random.RandomState(seed=random_seed)

        @utils.minibatch(batch_size, desc="fetch", use_last=True,
                         progress_bar=progress_bar)
        def _fetch(x, result):
            feed_dict = {self.training_flag: False}
            x = utils.densify(x)
            feed_dict[self.x] = x
            feed_dict[self.noisy_x] = random_state.poisson(x) if noisy else x
            # Tensorflow random samplers are fixed after creation,
            # making it impossible to re-seed and generate reproducible
            # results, so we use numpy samplers instead.
            # Also, local RandomState object is used to ensure thread-safety.
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)

        _fetch(x, result)
        return result

    def inference(self, x, batch_size=4096, noisy=0, progress_bar=False,
                  priority="auto", random_seed=config._USE_GLOBAL):
        """
        Project expression profiles into latent space.

        Parameters
        ----------
        x : array_like, or Cell_BLAST.data.ExprDataSet
            :math:`cell \\times gene` expression matrix.
        batch_size : int
            Minibatch size to use when doing projection, by default 4096.
            Changing this may slighly affect speed, but not inference result.
        noisy : int
            How many noisy samples to project, by default 0.
            If set to 0, no noise is added to the input during projection, so
            result is deterministic. If greater than 0, produces ``noisy``
            number of noisy space samples (posterior samples) for each cell.
        progress_bar : bool
            Whether to show progress bar duing projection, by default True.
        priority : {"auto", "speed", "memory"}
            Controls which one of speed or memory should be prioritized, by
            default "auto", meaning that data with more than 100,000 cells will
            use "memory" mode and smaller data will use "speed" mode.
        random_seed : int
            Random seed used with noisy projection. If not specified,
            ``Cell_BLAST.config.RANDOM_SEED`` will be used,
            which defaults to None.

        Returns
        -------
        latent : numpy.ndarray
            Coordinates in the latent space.
            If ``noisy`` is 0, will be in shape :math:`cell \\times latent\\_dim`.
            If ``noisy`` is greater than 0, will be in shape
            :math:`cell \\times noisy \\times latent\\_dim`.
        """
        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed
        if isinstance(x, data.ExprDataSet):
            x = x[:, self.genes].exprs
        if noisy > 0:
            if priority == "auto":
                priority = "memory" if x.shape[0] > 1e5 else "speed"
            if priority == "speed":
                xrep = [x] * noisy
                if scipy.sparse.issparse(x[0]):
                    xrep = scipy.sparse.vstack(xrep)
                else:
                    xrep = np.vstack(xrep)
                lrep = self._fetch(
                    self.latent, xrep, batch_size, True, progress_bar, random_seed)
                return np.stack(np.split(lrep, noisy), axis=1)
            else:  # priority == "memory":
                return np.stack([self._fetch(
                    self.latent, x, batch_size, True, progress_bar, random_seed
                ) for _ in range(noisy)], axis=1)
        return self._fetch(self.latent, x, batch_size, False, progress_bar)

    def clustering(self, x, batch_size=4096, progress_bar=True):
        """
        Get model intrinsic clustering of the data.

        Parameters
        ----------
        x : array_like or Cell_BLAST.data.ExprDataSet
            :math:`cell \\times gene` expression matrix.
        batch_size : int
            Minibatch size to use when getting clusters, by default 4096.
            Changing this may slighly affect speed, but not inference result.
        progress_bar : bool
            Whether to show progress bar during projection, by default True.

        Returns
        -------
        idx : numpy.ndarray
            model intrinsic clustering index, 1 dimensional
        confidence : numpy.ndarray
            model intrinsic clustering confidence, 1 dimensional
        """
        if not isinstance(self.latent_module, latent.CatGau):
            raise Exception("Model has no intrinsic clustering")
        if isinstance(x, data.ExprDataSet):
            x = x[:, self.genes].exprs
        cat = self._fetch(self.latent_module.cat, x, batch_size,
                          False, progress_bar)
        return cat.argmax(axis=1), cat.max(axis=1)

    def _save_weights(self, path):
        super(DIRECTi, self)._save_weights(os.path.join(path, "main"))
        with self.graph.as_default():
            self.latent_module._save_weights(
                self.sess, os.path.join(path, "latent"))
            self.prob_module._save_weights(
                self.sess, os.path.join(path, "prob"))
            for rmbatch_module in self.rmbatch_modules:
                rmbatch_module._save_weights(self.sess, os.path.join(
                    path, "rmbatch", rmbatch_module.name))

    def _load_weights(self, path, verbose=1):
        super(DIRECTi, self)._load_weights(os.path.join(path, "main"), verbose)
        with self.graph.as_default():
            message.info("Loading latent module weights...")
            self.latent_module._load_weights(self.sess, os.path.join(
                path, "latent"
            ), verbose, fast=self._mode == _TEST)
            if self._mode == _TEST:
                return
            message.info("Loading prob module weights...")
            self.prob_module._load_weights(self.sess, os.path.join(
                path, "prob"
            ), verbose)
            message.info("Loading rmbatch module weights...")
            for rmbatch_module in self.rmbatch_modules:
                rmbatch_module._load_weights(self.sess, os.path.join(
                    path, "rmbatch", rmbatch_module.name
                ), verbose)
            self.sess.run(tf.get_collection(tf.GraphKeys.READY_OP))

    def _get_config(self):
        config = {
            "genes": self.genes,
            "denoising": self.denoising,
            "decoder_feed_batch": self.decoder_feed_batch,
            "latent_module": self.latent_module._get_config(),
            "prob_module": self.prob_module._get_config(),
            "rmbatch_modules": [],
            **super(DIRECTi, self)._get_config()
        }
        for rmbatch_module in self.rmbatch_modules:
            config["rmbatch_modules"].append(rmbatch_module._get_config())
        return config

    @classmethod
    def _load_config(cls, file, **kwargs):
        with open(file, "r") as f:
            config = json.load(f)
        rmbatch_modules = []
        if "rmbatch_modules" in config:
            for rmbatch_module in config["rmbatch_modules"]:
                rmbatch_modules.append(
                    rmbatch.RMBatch._load_config(rmbatch_module)
                )
        model = cls(
            genes=config["genes"],
            denoising=config["denoising"],
            decoder_feed_batch=config["decoder_feed_batch"],
            latent_module=latent.Latent._load_config(config["latent_module"]),
            prob_module=prob.ProbModel._load_config(config["prob_module"]),
            rmbatch_modules=rmbatch_modules,
            path=os.path.dirname(file), **kwargs
        )
        with model.graph.as_default():
            model.sess.run(tf.global_variables_initializer())
        return model


def fit_DIRECTi(
    dataset, genes=None, supervision=None, batch_effect=None,
    latent_dim=10, cat_dim=None, h_dim=128, depth=1,
    prob_module="NB", rmbatch_module="Adversarial",
    latent_module_kwargs=None, prob_module_kwargs=None, rmbatch_module_kwargs=None,
    optimizer="RMSPropOptimizer", learning_rate=1e-3, batch_size=128,
    val_split=0.1, epoch=1000, patience=30, progress_bar=False,
    reuse_weights=None, random_seed=config._USE_GLOBAL, path="."
):
    """
    A convenient one-step function to build and fit DIRECTi models.
    Should work well in most cases.

    Parameters
    ----------
    dataset : Cell_BLAST.data.ExprDataSet
        Dataset to be fitted.
    genes : array_like
        Genes to use in the model, should be a subset of ``dataset.var_names``,
        by default None, which means all genes are used.
    supervision : str
        Specifies a character column in ``dataset.obs``, for use as
        (semi-)supervision, default is None.
        If value in the specified column is an emtpy string, corresponding
        cells will be treated as unsupervised.
    batch_effect : str, list
        Specifies one or more character columns in ``dataset.obs``
        for use as batch effect to be removed, default is None.
    latent_dim : int
        Latent space dimensionality, by default 10.
    cat_dim : int
        Number of intrinsic clusters, by default None.
        Note the difference that if not specified (None), only the continuous
        Gaussian latent is used. If set to 1, a single intrinsic cluster is
        used. Despite the different model structure, performance should be
        similar.
    h_dim : int
        Hidden layer dimensionality, by default 128.
        It is used consistently across all MLPs in the model.
    depth : int
        Hidden layer depth, by default 1.
        It is used consistently across all MLPs in the model.
    prob_module : {"NB", "ZINB", "ZIG"}
        Probabilistic model to fit, by default "NB". See `Cell_BLAST.prob`
        module for details.
    rmbatch_module : str, list
        Batch effect removing method, by default "Adversarial". If a list
        is specified, each element specifies the method used for a corresponding
        batch effect in ``batch_effect`` list.
    latent_module_kwargs : dict
        Keyword arguments to be passed to the latent module, by default None.
    prob_module_kwargs : dict
        Keyword arguments to be passed to the prob module, by default None.
    rmbatch_module_kwargs : dict, list
        Keyword arguments to be passed to the rmbatch module, by default None.
        If a list is specified, each element specifies keyword arguments
        for a corresponding batch effect in ``batch_effect`` list.
    optimizer : str
        Name of optimizer used in training, by default "RMSPropOptimizer".
    learning_rate : float
        Learning rate used in training, by default 1e-3.
    batch_size : int
        Size of minibatch used in training, by default 128.
    val_split : float
        Fraction of data to use for validation, by default 0.1.
    epoch : int
        Maximal training epochs, by default 1000.
    patience : int
        Early stop patience, by default 30.
        Model training stops when best validation loss does not decrease for
        ``patience`` epochs.
    progress_bar : bool
        Whether to show progress bar during training, by default False.
    reuse_weights : str
        Specifies a path where to reuse weights, by default None.
    random_seed : int
        Random seed. If not specified, ``Cell_BLAST.config.RANDOM_SEED``
        will be used, which defaults to None.
    path : str
        Specifies a path where model checkpoints as well as the final model
        is saved, by default ".".

    Returns
    -------
    model : DIRECTi
        A fitted DIRECTi model.

    Examples
    --------
    See the DIRECTi ipython notebook (:ref:`vignettes`) for live examples.
    """
    random_seed = config.RANDOM_SEED \
        if random_seed == config._USE_GLOBAL else random_seed
    if latent_module_kwargs is None:
        latent_module_kwargs = {}
    if prob_module_kwargs is None:
        prob_module_kwargs = {}
    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}

    data_dict = utils.DataDict()
    if genes is not None:
        dataset = dataset[:, genes]
    else:
        genes = dataset.var_names.values
    data_dict["exprs"] = dataset.exprs

    if batch_effect is None:
        batch_effect = []
    elif isinstance(batch_effect, str):
        batch_effect = [batch_effect]
    for _batch_effect in batch_effect:
        data_dict[_batch_effect] = utils.encode_onehot(
            dataset.obs[_batch_effect].astype(object).fillna("IgNoRe"),
            sort=True, ignore="IgNoRe"
        )  # sorting ensures batch order reproducibility for later tuning

    if supervision is not None:
        data_dict[supervision] = utils.encode_onehot(
            dataset.obs[supervision].astype(object).fillna("IgNoRe"),
            sort=True, ignore="IgNoRe"
        )  # sorting ensures supervision order reproducibility for later tuning
        if cat_dim is None:
            cat_dim = data_dict[supervision].shape[1]
        elif cat_dim > data_dict[supervision].shape[1]:
            data_dict[supervision] = scipy.sparse.hstack([
                data_dict[supervision].tocsc(),
                scipy.sparse.csc_matrix((
                    data_dict[supervision].shape[0],
                    cat_dim - data_dict[supervision].shape[1]
                ))
            ]).tocsr()
        elif cat_dim < data_dict[supervision].shape[1]:  # pragma: no cover
            raise ValueError(
                "`cat_dim` must be greater than or equal to "
                "number of supervised classes!"
            )
        # else ==

    kwargs = dict(latent_dim=latent_dim, h_dim=h_dim, depth=depth)
    if cat_dim:
        kwargs.update(dict(cat_dim=cat_dim))
        if supervision:
            kwargs.update(dict(name=supervision))
            kwargs.update(latent_module_kwargs)
            latent_module = latent.SemiSupervisedCatGau(**kwargs)
        else:
            kwargs.update(latent_module_kwargs)
            latent_module = latent.CatGau(**kwargs)
    else:
        kwargs.update(latent_module_kwargs)
        latent_module = latent.Gau(**kwargs)

    if not isinstance(rmbatch_module, list):
        rmbatch_module = [rmbatch_module] * len(batch_effect)
    if not isinstance(rmbatch_module_kwargs, list):
        rmbatch_module_kwargs = [rmbatch_module_kwargs] * len(batch_effect)
    assert len(rmbatch_module_kwargs) == len(rmbatch_module) == len(batch_effect)

    rmbatch_list = []
    for _batch_effect, _rmbatch_module, _rmbatch_module_kwargs in zip(
        batch_effect, rmbatch_module, rmbatch_module_kwargs
    ):
        kwargs = dict(
            batch_dim=data_dict[_batch_effect].shape[1],
            name=_batch_effect
        )
        if _rmbatch_module in (
            "Adversarial", "MNNAdversarial", "AdaptiveMNNAdversarial"
        ):
            kwargs.update(dict(h_dim=h_dim, depth=depth))
            kwargs.update(_rmbatch_module_kwargs)
        elif _rmbatch_module not in ("RMBatch", "MNN"):  # pragma: no cover
            raise ValueError("Invalid rmbatch method!")
        # else "RMBatch" or "MNN"
        kwargs.update(_rmbatch_module_kwargs)
        rmbatch_list.append(rmbatch.__dict__[_rmbatch_module](**kwargs))

    kwargs = dict(h_dim=h_dim, depth=depth)
    kwargs.update(prob_module_kwargs)
    prob_module = prob.__dict__[prob_module](**kwargs)
    model = DIRECTi(
        genes=genes,
        latent_module=latent_module,
        prob_module=prob_module,
        rmbatch_modules=rmbatch_list,
        path=path,
        random_seed=random_seed
    )
    model.compile(optimizer=optimizer, lr=learning_rate)
    if reuse_weights is not None:
        model._load_weights(reuse_weights, verbose=2)
    model.fit(
        data_dict, batch_size=batch_size, val_split=val_split,
        epoch=epoch, patience=patience, progress_bar=progress_bar
    )
    return model


def align_DIRECTi(
    model, original_dataset, new_dataset,
    rmbatch_module="MNNAdversarial", rmbatch_module_kwargs=None,
    deviation_reg=0.01, batch_size=256, val_split=0.1,
    optimizer="RMSPropOptimizer", learning_rate=1e-3,
    epoch=100, patience=100, reuse_weights=True,
    progress_bar=False, random_seed=config._USE_GLOBAL, path="."
):
    """
    Align datasets starting with an existing DIRECTi model

    Parameters
    ----------
    model : Cell_BLAST.directi.DIRECTi
        Pretrained model.
    original_dataset : Cell_BLAST.directi.ExprDataSet
        Dataset that the model was originally trained on.
    new_dataset : Cell_BLAST.data.ExprDataSet, dict
        New dataset or dictionary containing new datasets, to be aligned with
        ``original_dataset``.
    rmbatch_module: str
        Specifies the systematical bias / batch effect removing method to use
        for aligning new datasets, by default "MNNAdversarial".
    rmbatch_module_kwargs : dict
        Keyword arguments to be passed to rmbatch module, by default None.
    deviation_reg : float
        Regularization strength for deviation from original weights, by default
        0.01. Only applied to decoder (except for last layer which is fixed).
    batch_size : int
        Size of minibatch used in training, by default 256.
    val_split : float
        Fraction of data to use for validation, by default 0.1.
    optimizer : str
        Name of the optimizer to use, by default "RMSPropOptimizer".
    learning_rate : float
        Learning rate, by default 1e-3.
    epoch : int
        Maximal training epochs, by default 100.
    patience : int
        Early stop patience, by default 100. Model training stops when best
        validation loss does not decrease for a consecutive ``patience`` epochs.
    reuse_weights : bool
        Whether to reuse weights of the input model, by default True.
    progress_bar : bool
        Whether to show progress bar during training, by default False.
    random_seed : int
        Random seed. If not specified, ``Cell_BLAST.config.RANDOM_SEED``
        will be used, which defaults to None.
    path : str
        Specifies a path where model checkpoints as well as the final model
        is saved, by default ".".

    Returns
    -------
    aligned_model : Cell_BLAST.directi.DIRECTi
        Aligned model.
    """
    random_seed = config.RANDOM_SEED \
        if random_seed == config._USE_GLOBAL else random_seed
    if not os.path.exists(path):
        os.makedirs(path)
    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}
    if isinstance(new_dataset, data.ExprDataSet):
        new_datasets = {"new_dataset": new_dataset}
    else:
        assert isinstance(new_dataset, dict)
        new_datasets = new_dataset

    _config = model._get_config()
    for _rmbatch_module in _config["rmbatch_modules"]:
        _rmbatch_module["delay"] = 0
    kwargs = {
        "class": "Cell_BLAST.rmbatch.%s" % rmbatch_module,
        "batch_dim": len(new_datasets) + 1,
        "delay": 0, "name": "align"
    }
    if rmbatch_module in (
        "Adversarial", "MNNAdversarial", "AdaptiveMNNAdversarial"
    ):
        kwargs.update(dict(
            h_dim=model.latent_module.h_dim,
            depth=model.latent_module.depth,
            dropout=model.latent_module.dropout,
            lambda_reg=0.01
        ))
    elif rmbatch_module not in ("RMBatch", "MNN"):  # pragma: no cover
        raise ValueError("Unknown rmbatch_module!")
    # else "RMBatch" or "MNN"
    kwargs.update(rmbatch_module_kwargs)
    _config["rmbatch_modules"].append(kwargs)

    _config["prob_module"]["fine_tune"] = True
    _config["prob_module"]["deviation_reg"] = deviation_reg

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_config, f, indent=4)
    aligned_model = DIRECTi._load_config(
        os.path.join(path, "config.json"), random_seed=random_seed
    ).compile(optimizer, learning_rate)
    if reuse_weights:
        model._save_weights(os.path.join(path, "unaligned"))
        aligned_model._load_weights(os.path.join(path, "unaligned"), verbose=2)
    supervision = aligned_model.latent_module.name if isinstance(
        aligned_model.latent_module, latent.SemiSupervisedCatGau
    ) else None

    original_dataset = original_dataset[:, model.genes]
    assert "align" not in original_dataset.obs.columns
    for key in new_datasets.keys():
        assert "align" not in new_datasets[key].obs.columns
        new_datasets[key] = new_datasets[key][:, model.genes]
        new_datasets[key].obs = new_datasets[key].obs.loc[:, []]
        # All meta in new datasets are cleared to avoid interference
    assert "original" not in new_datasets
    dataset = {"original": original_dataset, **new_datasets}
    dataset = data.ExprDataSet.merge_datasets(dataset, meta_col="align")
    dataset = dataset[:, model.genes]  # merge_datasets can change gene order

    data_dict = utils.DataDict(exprs=dataset.exprs)
    for rmbatch_module in aligned_model.rmbatch_modules:
        data_dict[rmbatch_module.name] = utils.encode_onehot(
            dataset.obs[rmbatch_module.name].astype(object).fillna("IgNoRe"),
            sort=True, ignore="IgNoRe"
        )
    if isinstance(aligned_model.latent_module, latent.SemiSupervisedCatGau):
        data_dict[supervision] = utils.encode_onehot(
            dataset.obs[supervision].astype(object).fillna("IgNoRe"),
            sort=True, ignore="IgNoRe"
        )
        cat_dim = aligned_model.latent_module.cat_dim
        if cat_dim > data_dict[supervision].shape[1]:
            data_dict[supervision] = scipy.sparse.hstack([
                data_dict[supervision].tocsc(),
                scipy.sparse.csc_matrix((
                    data_dict[supervision].shape[0],
                    cat_dim - data_dict[supervision].shape[1]
                ))
            ]).tocsr()

    aligned_model.fit(
        data_dict, batch_size=batch_size, val_split=val_split,
        epoch=epoch, patience=patience, progress_bar=progress_bar
    )
    return aligned_model
