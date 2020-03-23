r"""
DIRECTi, an deep learning model for semi-supervised parametric dimension
reduction and systematical bias removal, extended from scVI.
"""

import json
import os
import typing
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse
import tensorflow as tf

from . import config, data, latent, model, prob, rmbatch, utils

_TRAIN = 1
_TEST = 0


class DIRECTi(model.Model):
    r"""
    DIRECTi model.

    Parameters
    ----------
    genes
        Genes to use in the model.
    latent_module
        Module for latent variable (encoder module).
    prob_module
        Module for data generative modeling (decoder module).
    rmbatch_modules
        List of modules for batch effect correction.
    denoising
        Whether to add noise to the input during training (source of randomness
        in modeling the approximate posterior).
    decoder_feed_batch
        How to feed batch information to the decoder.
        Available options are listed below:
        "nonlinear": concatenate with the cell embedding vector and go through
        all nonlinear transformations in the decoder;
        "linear": concatenate with last hidden layer in the decoder and only go
        through the last linear transformation;
        "both": concatenate with both the cell embedding vector and the last
        encoder hidden layer;
        False: do not feed batch information to the decoder.
    path
        Specifies a path where model configuration, checkpoints,
        as well as the final model will be saved.
    random_seed
        Random seed. If not specified, :data:`config.RANDOM_SEED`
        will be used, which defaults to 0.

    Attributes
    ----------
    genes
        List of gene names the model is defined and fitted on

    Examples
    --------

    The :func:`fit_DIRECTi` function offers an easy to use wrapper of this
    :class:`DIRECTi` model class, which is the preferred API and should satisfy most
    needs. We suggest using the :func:`fit_DIRECTi` wrapper first.
    However, if you do wish to use this :class:`DIRECTi` class,
    here's a brief instruction:

    First you need to select the proper modules required to build a complete model.
    :class:`DIRECTi` is made up of three types of modules, a latent (encoder) module,
    a generative (decoder) module and optional batch effect correction modules:

    >>> latent_module = Cell_BLAST.latent.Gau(latent_dim=10)
    >>> prob_module = Cell_BLAST.prob.ZINB()
    >>> rmbatch_modules = [Cell_BLAST.rmbatch.Adversarial(
    ...     batch_dim=2, name="rmbatch"
    ... )]

    We also need a list of gene names which defines the gene set on which
    the model will be fitted. It can also be accessed later to help ensure
    correct gene set of the input data.
    Then we can assemble them into a complete DIRECTi model, and compile it
    (get ready for model fitting):

    >>> model = Cell_BLAST.directi.DIRECTi(
    ...     genes, latent_module, prob_module, rmbatch_modules
    ... ).compile(optimizer="RMSPropOptimizer", lr=1e-3)

    To fit the model, we need a :class:`utils.DataDict` object,
    which can be seen as a dict of array-like objects.

    In the :class:`utils.DataDict`, we require an "exprs" slot that stores the
    :math:`cell \times gene` expression matrix.
    If the latent module :class:`latent.SemiSupervisedCatGau` is used, we additionally
    require a slot containing one-hot encoded supervision labels. Unsupervised
    cells should have all zeros in the corresponding rows. Name of the slot
    should be the same as name of the semi-supervision latent module.
    If batch effect correction modules are used, we additionally require slots
    containing one-hot encoded batch information. Name of the slots should be
    the same as name of batch correction modules.

    Here's an example on how to construct a data dict from a
    :class:`data.ExprDataSet` object:

    >>> data_dict = Cell_BLAST.utils.DataDict(
    ...     exprs=data_obj[:, model.genes].exprs,
    ...     rmbatch=Cell_BLAST.utils.encode_onehot(data_obj.obs["batch"])
    ... )

    Now we are ready to fit the model:

    >>> model.fit(data_dict)

    At this point we have obtained a fitted DIRECTi model object (which is also
    the return value if the :func:`fit_DIRECTi` function).
    We can use this model to project transriptomes into the low dimensional
    cell embedding space by using the :meth:`DIRECTi.inference` method.
    You may pass it to :attr:`data.ExprDataSet.latent` slot in the original
    data object, which facilitates subsequent visualization of the latent
    embedding space:

    >>> data_obj.latent = model.inference(data_obj)
    >>> data_obj.visualize_latent("cell_type")
    """

    _TRAIN = 1
    _TEST = 0

    def __init__(
            self, genes: typing.List[str],
            latent_module: latent.Latent,
            prob_module: prob.ProbModel,
            rmbatch_modules: typing.Optional[typing.List[rmbatch.RMBatch]] = None,
            denoising: bool = True,
            decoder_feed_batch: typing.Union[str, bool] = "nonlinear",
            path: typing.Optional[str] = None,
            random_seed: int = config._USE_GLOBAL,
            _mode: int = _TRAIN
    ) -> None:
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
            self, latent_module: latent.Latent, prob_module: prob.ProbModel,
            rmbatch_modules: typing.Optional[typing.List[rmbatch.RMBatch]] = None,
            denoising: bool = True,
            decoder_feed_batch: typing.Union[str, bool] = "nonlinear"
    ) -> None:
        super(DIRECTi, self)._init_graph()
        self.denoising = denoising
        self.decoder_feed_batch = decoder_feed_batch
        self.latent_module = latent_module
        self.prob_module = prob_module
        self.rmbatch_modules = rmbatch_modules or []
        self.loss, self.early_stop_loss = [], []

        with tf.name_scope("placeholder/"):
            self.x = tf.placeholder(
                dtype=tf.float32, shape=(None, self.x_dim), name="x")
            self.library_size = tf.placeholder(
                dtype=tf.float32, shape=(None, 1), name="library_size")
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag")

        # Preprocessing
        with tf.name_scope("normalize"):
            normalized_x = prob_module._normalize(self.x, self.library_size)
        with tf.name_scope("noise"):
            self.noisy_x = prob_module._add_noise(normalized_x) \
                if denoising else normalized_x
        with tf.name_scope("preprocess"):
            self.preprocessed_x = prob_module._preprocess(self.noisy_x)

        # Encoder
        self.latent = self.latent_module._build_latent(
            self.preprocessed_x, self.training_flag, scope="encoder")
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
        full_latent = [self.latent] + feed_batch if decoder_feed_batch in (
            "nonlinear", "both"
        ) and self.rmbatch_modules else [self.latent]
        tail_concat = feed_batch if decoder_feed_batch in (
            "linear", "both"
        ) and self.rmbatch_modules else None
        self.recon_loss = self.prob_module._loss(
            self.x, full_latent, self.training_flag,
            tail_concat=tail_concat, scope="decoder"
        )

        self.loss.append(self.recon_loss)
        self.early_stop_loss.append(self.recon_loss)

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

        self.grad_dict = {}

    def _compile(
            self, optimizer: str = "RMSPropOptimizer", lr: float = 1e-3
    ) -> None:
        if self.latent_module:
            self.latent_module._compile(optimizer, lr)
        if self.prob_module:
            self.prob_module._compile(optimizer, lr)
        for rmbatch_module in self.rmbatch_modules:
            rmbatch_module._compile(optimizer, lr)
        with tf.variable_scope("optimize/main"):
            control_dependencies = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(control_dependencies):
                self.step = getattr(tf.train, optimizer)(lr).minimize(
                    self.loss, var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "encoder"
                    ) + tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "decoder"
                    )
                )

    def _fit_epoch(
            self, data_dict: utils.DataDict, batch_size: int = 128,
            progress_bar: bool = True
    ) -> None:
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
                self.library_size: data_dict["library_size"],
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
        self.epoch_report += f"train={loss_dict[self.early_stop_loss]:.3f}, "

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag=f"{item.name} (train)",
                             simple_value=loss_dict[item])
            for item in loss_dict
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(
            self, data_dict: utils.DataDict, batch_size: int = 128,
            progress_bar: bool = True
    ) -> float:
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
                self.library_size: data_dict["library_size"],
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
        self.epoch_report += f"val={loss_dict[self.early_stop_loss]:.3f}, "

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag=f"{item.name} (val)",
                             simple_value=loss_dict[item])
            for item in loss_dict
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss_dict[self.early_stop_loss]

    def fit(
            self, data_dict: utils.DataDict, batch_size: int = 128,
            val_split: float = 0.1, epoch: int = 1000,
            patience: int = 30, tolerance: float = 0.0,
            on_epoch_end: typing.Optional[typing.List[typing.Callable[
                ["DIRECTi", utils.DataDict, utils.DataDict, float], bool
            ]]] = None,
            progress_bar: bool = False
    ) -> "DIRECTi":
        r"""
        Fit the model.

        Parameters
        ----------
        data_dict
            Training data.
        batch_size
            Size of minibatch used in training.
        val_split
            Fraction of data to use for validation.
        epoch
            Maximal training epochs.
        patience
            Early stop patience. Model training stops when the
            best validation loss does not decrease for a consecutive
            ``patience`` epochs.
        on_epoch_end
            List of functions to be executed at the end of each epoch.
        progress_bar
            Whether to print progress bar for each epoch during training.

        Returns
        -------
        model
            The fitted model
        """
        on_epoch_end = on_epoch_end or []
        on_epoch_end += \
            self.latent_module.on_epoch_end + self.prob_module.on_epoch_end
        for rmbatch_module in self.rmbatch_modules:
            on_epoch_end += rmbatch_module.on_epoch_end
        return super(DIRECTi, self).fit(
            data_dict, batch_size=batch_size, val_split=val_split, epoch=epoch,
            patience=patience, tolerance=tolerance, on_epoch_end=on_epoch_end,
            progress_bar=progress_bar
        )

    @utils.with_self_graph
    def _fetch(
            self, tensor: tf.Tensor,
            data_dict: typing.Optional[utils.DataDict] = None,
            batch_size: int = 4096, noisy: bool = False,
            progress_bar: bool = False, random_seed: int = config._USE_GLOBAL
    ) -> np.ndarray:
        if data_dict is None:
            return self.sess.run(tensor)
        if noisy:
            random_seed = config.RANDOM_SEED \
                if random_seed == config._USE_GLOBAL else random_seed
            random_state = np.random.RandomState(seed=random_seed)
        result_shape = tensor.get_shape().as_list()
        if result_shape[0] is None:
            result_shape[0] = data_dict.shape[0]
        result = np.empty(result_shape)

        @utils.minibatch(batch_size, desc="fetch", use_last=True,
                         progress_bar=progress_bar)
        def _fetch_minibatch(data_dict, result):
            feed_dict = {self.training_flag: False}
            if "exprs" in data_dict and "library_size" in data_dict:
                x = data_dict["exprs"]
                normalized_x = self.prob_module._normalize(x, data_dict["library_size"])
                feed_dict.update({
                    self.x: utils.densify(x),
                    self.noisy_x: self.prob_module._add_noise(
                        utils.densify(normalized_x), random_state
                    ) if noisy else utils.densify(normalized_x),
                    self.library_size: data_dict["library_size"]
                })
                # Tensorflow random samplers are fixed after creation,
                # making it impossible to re-seed and generate reproducible
                # results, so we use numpy samplers instead.
                # Also, local RandomState object is used to ensure thread-safety.
            for module in [self.latent_module, self.prob_module, *self.rmbatch_modules]:
                try:
                    feed_dict.update(module._build_feed_dict(data_dict))
                except Exception:
                    pass
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)

        _fetch_minibatch(data_dict, result)
        return result

    def inference(
            self, dataset: data.ExprDataSet, batch_size: int = 4096,
            n_posterior: int = 0, progress_bar: bool = False,
            priority: str = "auto", random_seed: int = config._USE_GLOBAL
    ) -> np.ndarray:
        r"""
        Project expression profiles into the cell embedding space.

        Parameters
        ----------
        x
            Dataset for which to compute cell embeddings.
        batch_size
            Minibatch size.
            Changing this may slighly affect speed, but not the result.
        n_posterior
            How many posterior samples to fetch.
            If set to 0, the posterior point estimate is computed.
            If greater than 0, produces ``n_posterior`` number of
            posterior samples for each cell.
        progress_bar
            Whether to show progress bar duing projection.
        priority
            Should be among {"auto", "speed", "memory"}.
            Controls which one of speed or memory should be prioritized, by
            default "auto", meaning that data with more than 100,000 cells will
            use "memory" mode and smaller data will use "speed" mode.
        random_seed
            Random seed used with noisy projection. If not specified,
            :data:`config.RANDOM_SEED` will be used, which defaults to 0.

        Returns
        -------
        latent
            Coordinates in the latent space.
            If ``n_posterior`` is 0, will be in shape :math:`cell \times latent\_dim`.
            If ``n_posterior`` is greater than 0, will be in shape
            :math:`cell \times noisy \times latent\_dim`.
        """
        random_seed = config.RANDOM_SEED \
            if random_seed == config._USE_GLOBAL else random_seed
        x = dataset[:, self.genes].exprs
        l = np.array(dataset.exprs.sum(axis=1)).reshape((-1, 1)) \
            if "__libsize__" not in dataset.obs.columns \
            else dataset.obs["__libsize__"].values.reshape((-1, 1))
        if n_posterior > 0:
            if priority == "auto":
                priority = "memory" if x.shape[0] > 1e4 else "speed"
            if priority == "speed":
                if scipy.sparse.issparse(x):
                    xrep = x.tocsr()[np.repeat(np.arange(x.shape[0]), n_posterior)]
                else:
                    xrep = np.repeat(x, n_posterior, axis=0)
                lrep = np.repeat(l, n_posterior, axis=0)
                data_dict = utils.DataDict(exprs=xrep, library_size=lrep)
                return self._fetch(
                    self.latent, data_dict, batch_size, True, progress_bar, random_seed
                ).astype(np.float32).reshape((x.shape[0], n_posterior, -1))
            else:  # priority == "memory":
                data_dict = utils.DataDict(exprs=x, library_size=l)
                return np.stack([self._fetch(
                    self.latent, data_dict, batch_size, True, progress_bar,
                    (random_seed + i) if random_seed is not None else None
                ).astype(np.float32) for i in range(n_posterior)], axis=1)
        data_dict = utils.DataDict(exprs=x, library_size=l)
        return self._fetch(
            self.latent, data_dict, batch_size, False, progress_bar
        ).astype(np.float32)

    def clustering(
            self, dataset: data.ExprDataSet, batch_size: int = 4096,
            return_confidence: bool = False, progress_bar: bool = False
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        r"""
        Get model intrinsic clustering of the data.

        Parameters
        ----------
        x
            Dataset for which to obtain the intrinsic clustering.
        batch_size
            Minibatch size.
            Changing this may slighly affect speed, but not the result.
        progress_bar
            Whether to show progress bar during projection.

        Returns
        -------
        idx
            model intrinsic clustering index, 1 dimensional
        confidence
            model intrinsic clustering confidence, 1 dimensional
        """
        if not isinstance(self.latent_module, latent.CatGau):
            raise Exception("Model has no intrinsic clustering")
        x = dataset[:, self.genes].exprs
        l = np.array(dataset.exprs.sum(axis=1)).reshape((-1, 1)) \
            if "__libsize__" not in dataset.obs.columns \
            else dataset.obs["__libsize__"].values.reshape((-1, 1))
        data_dict = utils.DataDict(exprs=x, library_size=l)
        cat = self._fetch(
            self.latent_module.cat, data_dict, batch_size, False, progress_bar
        ).astype(np.float32)
        if return_confidence:
            return cat.argmax(axis=1), cat.max(axis=1)
        return cat.argmax(axis=1)

    @utils.with_self_graph
    def _fetch_grad(
            self, input_tensor: tf.Tensor, output_tensor: tf.Tensor,
            data_dict: utils.DataDict, batch_size: int = 4096,
            progress_bar: bool = False
    ) -> np.ndarray:
        r"""
        Requires "output_grad" slot in data_dict as the gradient source.
        Additionally, it requires either explicit value of the output tensor (as
        "output" slot in data_dict), or sufficient data to compute the output
        tensor (e.g., "exprs" and "library_size" slots if output tensor
        is the latent variable).
        """
        input_scope_safe_name = utils.scope_free(input_tensor.name)
        output_scope_safe_name = utils.scope_free(output_tensor.name)
        with tf.name_scope("custom_grad/"):
            if output_tensor not in self.grad_dict:
                self.grad_dict[output_tensor] = tf.placeholder(
                    dtype=tf.float32,
                    shape=output_tensor.shape,
                    name=f"{output_scope_safe_name}_grad"
                )
            if (input_tensor, output_tensor) not in self.grad_dict:
                self.grad_dict[(input_tensor, output_tensor)] = tf.gradients(
                    output_tensor, input_tensor,
                    grad_ys=self.grad_dict[output_tensor],
                    name=f"{input_scope_safe_name}_grad_from_{output_scope_safe_name}"
                )[0]

        result_shape = input_tensor.get_shape().as_list()
        if result_shape[0] is None:
            result_shape[0] = data_dict.shape[0]
        result = np.empty(result_shape)

        @utils.minibatch(batch_size, desc="fetch_grad", use_last=True,
                         progress_bar=progress_bar)
        def _fetch_grad_minibatch(data_dict, result):
            feed_dict = {
                self.grad_dict[output_tensor]: data_dict["output_grad"],
                self.training_flag: False
            }
            if "output" in data_dict:
                feed_dict[output_tensor] = data_dict["output"]
            if "exprs" in data_dict and "library_size" in data_dict:
                x = data_dict["exprs"]
                normalized_x = self.prob_module._normalize(x, data_dict["library_size"])
                feed_dict.update({
                    self.x: utils.densify(x),
                    self.noisy_x: utils.densify(normalized_x),
                    self.library_size: data_dict["library_size"]
                })
            for module in [self.latent_module, self.prob_module, *self.rmbatch_modules]:
                try:
                    feed_dict.update(module._build_feed_dict(data_dict))
                except Exception:
                    pass
            result[:] = self.sess.run(
                self.grad_dict[(input_tensor, output_tensor)],
                feed_dict=feed_dict
            )

        _fetch_grad_minibatch(data_dict, result)
        return result

    @utils.with_self_graph
    def gene_grad(
            self, dataset: data.ExprDataSet, latent_grad: np.ndarray,
            batch_size: int = 4096, progress_bar: bool = False
    ) -> np.ndarray:
        r"""
        Fetch gene space gradients with regard to latent space gradients

        Parameters
        ----------
        dataset
            Dataset for which to obtain gene gradients.
        latent_grad
            Latent space gradients.
        batch_size
            Minibatch size.
            Changing this may slighly affect speed, but not the result.
        progress_bar
            Whether to show progress bar during projection.

        Returns
        -------
        grad
            Fetched gene-wise gradient
        """
        x = dataset[:, self.genes].exprs
        l = np.array(dataset.exprs.sum(axis=1)).reshape((-1, 1)) \
            if "__libsize__" not in dataset.obs.columns \
            else dataset.obs["__libsize__"].values.reshape((-1, 1))
        data_dict = utils.DataDict(
            exprs=x, library_size=l, output_grad=latent_grad)
        return self._fetch_grad(
            self.preprocessed_x, self.latent, data_dict,
            batch_size=batch_size, progress_bar=progress_bar
        )

    def _save_weights(self, path: str) -> None:
        super(DIRECTi, self)._save_weights(os.path.join(path, "main"))
        with self.graph.as_default():  # pylint: disable=not-context-manager
            self.latent_module._save_weights(
                self.sess, os.path.join(path, "latent"))
            self.prob_module._save_weights(
                self.sess, os.path.join(path, "prob"))
            for rmbatch_module in self.rmbatch_modules:
                rmbatch_module._save_weights(self.sess, os.path.join(
                    path, "rmbatch", rmbatch_module.name))

    def _load_weights(self, path: str) -> None:
        super(DIRECTi, self)._load_weights(os.path.join(path, "main"))
        with self.graph.as_default():  # pylint: disable=not-context-manager
            utils.logger.info("Loading latent module weights...")
            self.latent_module._load_weights(self.sess, os.path.join(
                path, "latent"
            ), fast=self._mode == _TEST)
            if self._mode == _TEST:
                return
            utils.logger.info("Loading prob module weights...")
            self.prob_module._load_weights(self.sess, os.path.join(path, "prob"))
            utils.logger.info("Loading rmbatch module weights...")
            for rmbatch_module in self.rmbatch_modules:
                rmbatch_module._load_weights(self.sess, os.path.join(
                    path, "rmbatch", rmbatch_module.name
                ))
            self.sess.run(tf.get_collection(tf.GraphKeys.READY_OP))

    def _get_config(self) -> typing.Mapping:
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
    def _load_config(cls, file, **kwargs) -> "DIRECTi":
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
        dataset: data.ExprDataSet,
        genes: typing.Optional[typing.List[str]] = None,
        supervision: typing.Optional[str] = None,
        batch_effect: typing.Optional[typing.List[str]] = None,
        latent_dim: int = 10, cat_dim: typing.Optional[int] = None,
        h_dim: int = 128, depth: int = 1, prob_module: str = "NB",
        rmbatch_module: typing.Union[str, typing.List[str]] = "Adversarial",
        latent_module_kwargs: typing.Optional[typing.Mapping] = None,
        prob_module_kwargs: typing.Optional[typing.Mapping] = None,
        rmbatch_module_kwargs: typing.Optional[typing.Union[
            typing.Mapping, typing.List[typing.Mapping]
        ]] = None,
        optimizer: str = "RMSPropOptimizer", learning_rate: float = 1e-3,
        batch_size: int = 128, val_split: float = 0.1, epoch: int = 1000,
        patience: int = 30, progress_bar: bool = False, reuse_weights=None,
        random_seed: int = config._USE_GLOBAL, path: typing.Optional[str] = None
) -> DIRECTi:
    r"""
    A convenient one-step function to build and fit DIRECTi models.
    Should work well in most cases.

    Parameters
    ----------
    dataset
        Dataset to be fitted.
    genes
        Genes to fit on, should be a subset of :attr:`data.ExprDataSet.var_names`.
        If not specified, all genes are used.
    supervision
        Specifies a column in the :attr:`data.ExprDataSet.obs` table for use as
        (semi-)supervision. If value in the specified column is emtpy,
        the corresponding cells will be treated as unsupervised.
    batch_effect
        Specifies one or more columns in the :attr:`data.ExprDataSet.obs` table
        for use as batch effect to be corrected.
    latent_dim
        Latent space (cell embedding) dimensionality.
    cat_dim
        Number of intrinsic clusters.
    h_dim
        Hidden layer dimensionality. It is used consistently across all MLPs
        in the model.
    depth
        Hidden layer depth. It is used consistently across all MLPs in the model.
    prob_module
        Generative model to fit, should be among {"NB", "ZINB", "LN", "ZILN"}.
        See the :mod:`prob` for details.
    rmbatch_module
        Batch effect correction method. If a list is provided, each element
        specifies the method to use for a corresponding batch effect in
        ``batch_effect`` list (in this case the ``rmbatch_module`` list should
        have the same length as the ``batch_effect`` list).
    latent_module_kwargs
        Keyword arguments to be passed to the latent module.
    prob_module_kwargs
        Keyword arguments to be passed to the prob module.
    rmbatch_module_kwargs
        Keyword arguments to be passed to the rmbatch module.
        If a list is provided, each element specifies keyword arguments
        for a corresponding batch effect correction module in the
        ``rmbatch_module`` list.
    optimizer
        Name of optimizer used in training.
    learning_rate
        Learning rate used in training.
    batch_size
        Size of minibatch used in training.
    val_split
        Fraction of data to use for validation.
    epoch
        Maximal training epochs.
    patience
        Early stop patience. Model training stops when best validation loss does
        not decrease for a consecutive ``patience`` epochs.
    progress_bar
        Whether to show progress bars during training.
    reuse_weights
        Specifies a path where previously stored model weights can be reused.
    random_seed
        Random seed. If not specified, :data:`config.RANDOM_SEED`
        will be used, which defaults to 0.
    path
        Specifies a path where model checkpoints as well as the final model
        will be saved.

    Returns
    -------
    model
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

    if genes is None:
        genes = dataset.var_names.values
    data_dict = utils.DataDict(
        library_size=np.array(dataset.exprs.sum(axis=1)).reshape((-1, 1))
        if "__libsize__" not in dataset.obs.columns
        else dataset.obs["__libsize__"].values.reshape((-1, 1)),
        exprs=dataset[:, genes].exprs
    )

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
        rmbatch_list.append(getattr(rmbatch, _rmbatch_module)(**kwargs))

    kwargs = dict(h_dim=h_dim, depth=depth)
    kwargs.update(prob_module_kwargs)
    prob_module = getattr(prob, prob_module)(**kwargs)
    model = DIRECTi(
        genes=genes,
        latent_module=latent_module,
        prob_module=prob_module,
        rmbatch_modules=rmbatch_list,
        path=path,
        random_seed=random_seed
    ).compile(optimizer=optimizer, lr=learning_rate)
    if reuse_weights is not None:
        model._load_weights(reuse_weights)
    model.fit(
        data_dict, batch_size=batch_size, val_split=val_split,
        epoch=epoch, patience=patience, progress_bar=progress_bar
    )
    return model


def align_DIRECTi(
        model: DIRECTi, original_dataset: data.ExprDataSet,
        new_dataset: typing.Union[data.ExprDataSet, typing.Mapping[str, data.ExprDataSet]],
        rmbatch_module: str = "MNNAdversarial",
        rmbatch_module_kwargs: typing.Optional[typing.Mapping] = None,
        deviation_reg: float = 0.01, batch_size: int = 256, val_split: float = 0.1,
        optimizer: str = "RMSPropOptimizer", learning_rate: float = 1e-3,
        epoch: int = 100, patience: int = 100, tolerance: float = 0.0,
        reuse_weights: bool = True, progress_bar: bool = False,
        random_seed: int = config._USE_GLOBAL, path: typing.Optional[str] = None
) -> DIRECTi:
    r"""
    Align datasets starting with an existing DIRECTi model (fine-tuning)

    Parameters
    ----------
    model
        A pretrained DIRECTi model.
    original_dataset
        The dataset that the model was originally trained on.
    new_dataset
        A new dataset or a dictionary containing new datasets,
        to be aligned with ``original_dataset``.
    rmbatch_module
        Specifies the batch effect correction method to use for aligning new
        datasets.
    rmbatch_module_kwargs
        Keyword arguments to be passed to the rmbatch module.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    batch_size
        Size of minibatches used in training.
    val_split
        Fraction of data to use for validation.
    optimizer
        Name of the optimizer to use.
    learning_rate
        Learning rate.
    epoch
        Maximal training epochs.
    patience
        Early stop patience. Model training stops when best
        validation loss does not decrease for a consecutive ``patience`` epochs.
    tolerance
        Tolerance of deviation from the lowest validation loss recorded for the
        "patience countdown" to be reset. The "patience countdown" is reset if
        current validation loss < lowest validation loss recorded + ``tolerance``.
    reuse_weights
        Whether to reuse weights of the original model.
    progress_bar
        Whether to show progress bar during training.
    random_seed
        Random seed. If not specified, :data:`config.RANDOM_SEED`
        will be used, which defaults to 0.
    path
        Specifies a path where model checkpoints as well as the final model
        is saved.

    Returns
    -------
    aligned_model
        Aligned model.
    """
    random_seed = config.RANDOM_SEED \
        if random_seed == config._USE_GLOBAL else random_seed
    if path is None:
        path = tempfile.mkdtemp()
    else:
        os.makedirs(path, exist_ok=True)
    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}
    if isinstance(new_dataset, data.ExprDataSet):
        new_datasets = {"__new__": new_dataset}
    elif isinstance(new_dataset, dict):
        assert "__original__" not in new_dataset, \
            "Key `__original__` is now allowed in new datasets."
        new_datasets = new_dataset.copy()  # shallow
    else:
        raise TypeError("Invalid type for argument `new_dataset`.")

    _config = model._get_config()
    for _rmbatch_module in _config["rmbatch_modules"]:
        _rmbatch_module["delay"] = 0
    kwargs = {
        "class": f"Cell_BLAST.rmbatch.{rmbatch_module}",
        "batch_dim": len(new_datasets) + 1,
        "delay": 0, "name": "__align__"
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
        aligned_model._load_weights(os.path.join(path, "unaligned"))
    supervision = aligned_model.latent_module.name if isinstance(
        aligned_model.latent_module, latent.SemiSupervisedCatGau
    ) else None

    assert "__align__" not in original_dataset.obs.columns, \
        "Please remove column `__align__` from obs of the original dataset."
    original_dataset = original_dataset.copy()  # shallow
    original_dataset.obs = original_dataset.obs.copy(deep=False)
    if "__libsize__" not in original_dataset.obs.columns:
        original_dataset.obs["__libsize__"] = np.array(original_dataset.exprs.sum(axis=1)).ravel()
    original_dataset = original_dataset[:, model.genes]
    for key in new_datasets.keys():
        assert "__align__" not in new_datasets[key].obs.columns, \
            f"Please remove column `__align__` from new dataset {key}."
        new_datasets[key] = new_datasets[key].copy()  # shallow
        new_datasets[key].obs = new_datasets[key].obs.copy(deep=False)
        new_datasets[key].obs = new_datasets[key].obs.loc[:, new_datasets[key].obs.columns == "__libsize__"]
        if "__libsize__" not in new_datasets[key].obs.columns:
            new_datasets[key].obs["__libsize__"] = np.array(new_datasets[key].exprs.sum(axis=1)).ravel()
        new_datasets[key] = new_datasets[key][:, model.genes]
        # All meta in new datasets are cleared to avoid interference
    dataset = {"__original__": original_dataset, **new_datasets}
    dataset = data.ExprDataSet.merge_datasets(dataset, meta_col="__align__")

    data_dict = utils.DataDict(
        library_size=dataset.obs["__libsize__"].values.reshape((-1, 1)),
        exprs=dataset[:, model.genes].exprs  # Ensure order
    )
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
        epoch=epoch, patience=patience, tolerance=tolerance,
        progress_bar=progress_bar
    )
    return aligned_model
