r"""
DIRECTi, an deep learning model for semi-supervised parametric dimension
reduction and systematical bias removal, extended from scVI.
"""

import os
import tempfile
import time
import typing
from collections import OrderedDict

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributions as D
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from . import config, data, latent, prob, rmbatch, utils
from .config import DEVICE
from .rebuild import RMSprop

_TRAIN = 1
_TEST = 0


class DIRECTi(nn.Module):
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
    batch_effect
        Batch effects need to be corrected.
    rmbatch_modules
        List of modules for batch effect correction.
    denoising
        Whether to add noise to the input during training (source of randomness
        in modeling the approximate posterior).
    learning_rate
        Learning rate.
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
    batch_effect_list
        List of batch effect names need to be corrected.

    Examples
    --------
    The :func:`fit_DIRECTi` function offers an easy to use wrapper of this
    :class:`DIRECTi` model class, which is the preferred API and should satisfy most
    needs. We suggest using the :func:`fit_DIRECTi` wrapper first.

    """

    _TRAIN = 1
    _TEST = 0

    def __init__(
        self,
        genes: typing.List[str],
        latent_module: "latent.Latent",
        prob_module: "prob.ProbModel",
        rmbatch_modules: typing.Tuple["rmbatch.RMBatch"],
        denoising: bool = True,
        learning_rate: float = 1e-3,
        path: typing.Optional[str] = None,
        random_seed: int = config._USE_GLOBAL,
        _mode: int = _TRAIN,
    ) -> None:
        super().__init__()

        if path is None:
            path = tempfile.mkdtemp()
        random_seed = (
            config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed
        )
        self.ensure_reproducibility(random_seed)

        self.genes = genes
        self.latent_module = latent_module
        self.prob_module = prob_module
        self.rmbatch_modules = rmbatch_modules
        self.denoising = denoising
        self.learning_rate = learning_rate
        self.path = path
        self.random_seed = random_seed
        self._mode = _mode

        self.opt_latent_reg = RMSprop(
            self.latent_module.parameters_reg(), lr=learning_rate
        )
        self.opt_latent_fit = RMSprop(
            self.latent_module.parameters_fit(), lr=learning_rate
        )
        self.opt_prob = RMSprop(self.prob_module.parameters(), lr=learning_rate)
        self.opts_rmbatch = [
            RMSprop(_rmbatch.parameters(), lr=learning_rate)
            if _rmbatch._class
            in ("Adversarial", "MNNAdversarial", "AdaptiveMNNAdversarial")
            else None
            for _rmbatch in self.rmbatch_modules
        ]

    @staticmethod
    def ensure_reproducibility(random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    def get_config(self) -> typing.Mapping:
        return {
            "genes": self.genes,
            "latent_module": self.latent_module.get_config(),
            "prob_module": self.prob_module.get_config(),
            "rmbatch_modules": [
                _module.get_config() for _module in self.rmbatch_modules
            ],
            "denoising": self.denoising,
            "learning_rate": self.learning_rate,
            "path": self.path,
            "random_seed": self.random_seed,
            "_mode": self._mode,
        }

    @staticmethod
    def preprocess(
        x: torch.Tensor, libs: torch.Tensor, noisy: bool = True
    ) -> torch.Tensor:
        x = x / (libs / 10000)
        if noisy:
            x = D.Poisson(rate=x).sample()
        x = x.log1p()
        return x

    def fit(
        self,
        dataset: data.Dataset,
        batch_size: int = 128,
        val_split: float = 0.1,
        epoch: int = 1000,
        patience: int = 30,
        tolerance: float = 0.0,
        progress_bar: bool = False,
    ):
        os.makedirs(self.path, exist_ok=True)
        utils.logger.info("Using model path: %s", self.path)

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        assert self._mode == _TRAIN
        self.to(DEVICE)
        self.ensure_reproducibility(self.random_seed)
        self.save_weights(self.path)

        self.latent_module.check_fine_tune()
        self.prob_module.check_fine_tune()

        patience_remain = patience
        best_loss = 1e10

        summarywriter = SummaryWriter(log_dir=os.path.join(self.path, "summary"))

        for _epoch in range(epoch):
            start_time = time.time()

            if progress_bar:
                train_dataloader = utils.smart_tqdm()(train_dataloader)
            train_loss = self.train_epoch(train_dataloader, _epoch, summarywriter)

            if progress_bar:
                val_dataloader = utils.smart_tqdm()(val_dataloader)
            val_loss = self.val_epoch(val_dataloader, _epoch, summarywriter)

            report = f"[{self.__class__.__name__} epoch {_epoch}] "
            report += f"train={train_loss:.3f}, "
            report += f"val={val_loss:.3f}, "
            report += f"time elapsed={time.time() - start_time:.1f}s"

            if any([_epoch < _rmbatch.delay for _rmbatch in self.rmbatch_modules]):
                if _epoch % 10 == 0:
                    report += " Regular save..."
                    self.save_weights(self.path)
            elif val_loss < best_loss + tolerance:
                report += " Best save..."
                self.save_weights(self.path)
                best_loss = val_loss
                patience_remain = patience
            else:
                patience_remain -= 1

            print(report)

            if patience_remain < 0:
                break

        print("Restoring best model...")
        self.load_weights(self.path)
        self.save_weights(self.path)

    def train_epoch(self, train_dataloader, epoch, summarywriter):
        self.train()

        loss_record = {}
        self.latent_module.init_loss_record(loss_record)
        self.prob_module.init_loss_record(loss_record)
        for _rmbatch in self.rmbatch_modules:
            _rmbatch.init_loss_record(loss_record)
        loss_record["early_stop_loss"] = 0
        loss_record["total_loss"] = 0
        datasize = 0

        for feed_dict in train_dataloader:
            for key, value in feed_dict.items():
                feed_dict[key] = value.to(DEVICE)

            exprs = feed_dict["exprs"]
            libs = feed_dict["library_size"]
            datasize += libs.shape[0]

            x = self.preprocess(exprs, libs, self.denoising)
            l, l_components = self.latent_module(x)
            latent_d_loss = self.latent_module.d_loss(
                l_components, feed_dict, loss_record
            )
            self.opt_latent_reg.zero_grad()
            latent_d_loss.backward()
            self.opt_latent_reg.step()

            l = l.detach()
            for _rmbatch, _opt in zip(self.rmbatch_modules, self.opts_rmbatch):
                if epoch >= _rmbatch.delay:
                    mask = _rmbatch.get_mask(l, feed_dict)
                    if mask.sum() > 0:
                        for _ in range(_rmbatch.n_steps):
                            pred = _rmbatch(l, mask)
                            rmbatch_d_loss = _rmbatch.d_loss(
                                pred, feed_dict, mask, loss_record
                            )
                            if not _opt is None:
                                _opt.zero_grad()
                                rmbatch_d_loss.backward()
                                _opt.step()

            x = self.preprocess(exprs, libs, self.denoising)
            l, l_components = self.latent_module(x)
            latent_g_loss = self.latent_module.g_loss(
                l_components, feed_dict, loss_record
            )
            full_l = [l]
            for _rmbatch in self.rmbatch_modules:
                full_l.append(feed_dict[_rmbatch.name])
            d_components = self.prob_module(full_l, feed_dict)
            prob_loss = self.prob_module.loss(d_components, feed_dict, loss_record)
            loss = prob_loss + latent_g_loss
            for _rmbatch in self.rmbatch_modules:
                if epoch >= _rmbatch.delay:
                    mask = _rmbatch.get_mask(l, feed_dict)
                    if mask.sum() > 0:
                        pred = _rmbatch(l, mask)
                        rmbatch_g_loss = _rmbatch.g_loss(
                            pred, feed_dict, mask, loss_record
                        )
                        loss = loss + rmbatch_g_loss

            self.opt_latent_fit.zero_grad()
            self.opt_prob.zero_grad()
            loss.backward()
            self.opt_latent_fit.step()
            self.opt_prob.step()

            loss_record["early_stop_loss"] += prob_loss.item() * x.shape[0]
            loss_record["total_loss"] += loss.item() * x.shape[0]

        for key, value in loss_record.items():
            summarywriter.add_scalar(key + ":0 (train)", value / datasize, epoch)

        return loss_record["early_stop_loss"] / datasize

    def val_epoch(self, val_dataloader, epoch, summarywriter):
        self.eval()

        loss_record = {}
        self.latent_module.init_loss_record(loss_record)
        self.prob_module.init_loss_record(loss_record)
        for _rmbatch in self.rmbatch_modules:
            _rmbatch.init_loss_record(loss_record)
        loss_record["early_stop_loss"] = 0
        loss_record["total_loss"] = 0
        datasize = 0

        for feed_dict in val_dataloader:
            for key, value in feed_dict.items():
                feed_dict[key] = value.to(DEVICE)

            exprs = feed_dict["exprs"]
            libs = feed_dict["library_size"]
            datasize += libs.shape[0]

            with torch.no_grad():
                x = self.preprocess(exprs, libs, self.denoising)
                l, l_components = self.latent_module(x)
                _ = self.latent_module.d_loss(l_components, feed_dict, loss_record)

                for _rmbatch in self.rmbatch_modules:
                    if epoch >= _rmbatch.delay:
                        mask = _rmbatch.get_mask(l, feed_dict)
                        if mask.sum() > 0:
                            for _ in range(_rmbatch.n_steps):
                                pred = _rmbatch(l, mask)
                                _ = _rmbatch.d_loss(pred, feed_dict, mask, loss_record)

                x = self.preprocess(exprs, libs, self.denoising)
                l, l_components = self.latent_module(x)
                latent_g_loss = self.latent_module.g_loss(
                    l_components, feed_dict, loss_record
                )
                full_l = [l]
                for _rmbatch in self.rmbatch_modules:
                    full_l.append(feed_dict[_rmbatch.name])
                d_components = self.prob_module(full_l, feed_dict)
                prob_loss = self.prob_module.loss(d_components, feed_dict, loss_record)
                loss = prob_loss + latent_g_loss
                for _rmbatch in self.rmbatch_modules:
                    if epoch >= _rmbatch.delay:
                        mask = _rmbatch.get_mask(l, feed_dict)
                        if mask.sum() > 0:
                            pred = _rmbatch(l, mask)
                            rmbatch_g_loss = _rmbatch.g_loss(
                                pred, feed_dict, mask, loss_record
                            )
                            loss = loss + rmbatch_g_loss

            loss_record["early_stop_loss"] += prob_loss.item() * x.shape[0]
            loss_record["total_loss"] += loss.item() * x.shape[0]

        for key, value in loss_record.items():
            summarywriter.add_scalar(key + ":0 (val)", value / datasize, epoch)

        return loss_record["early_stop_loss"] / datasize

    def save_weights(self, path: str, checkpoint: str = "checkpoint.pk"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, checkpoint))

    def load_weights(self, path: str, checkpoint: str = "checkpoint.pk"):
        assert os.path.exists(path)
        self.load_state_dict(torch.load(os.path.join(path, checkpoint), map_location=DEVICE))

    @classmethod
    def load_config(cls, configuration: typing.Mapping):
        _class = configuration["latent_module"]["_class"]
        latent_module = getattr(latent, _class)(**configuration["latent_module"])

        _class = configuration["prob_module"]["_class"]
        prob_module = getattr(prob, _class)(**configuration["prob_module"])

        rmbatch_modules = nn.ModuleList()
        for _conf in configuration["rmbatch_modules"]:
            _class = _conf["_class"]
            rmbatch_modules.append(getattr(rmbatch, _class)(**_conf))

        configuration["latent_module"] = latent_module
        configuration["prob_module"] = prob_module
        configuration["rmbatch_modules"] = rmbatch_modules

        model = cls(**configuration)

        return model

    def save(
        self,
        path: typing.Optional[str] = None,
        config: str = "config.pk",
        weights: str = "weights.pk",
    ):
        r"""
        Save model to files

        Parameters
        ----------
        path
            Path to a directory where the model will be saved
        config
            Name of the configuration file
        weights
            Name of the weights file
        """
        if path is None:
            os.makedirs(self.path, exist_ok=True)
            torch.save(self.get_config(), os.path.join(self.path, config))
            torch.save(self.state_dict(), os.path.join(self.path, weights))
        else:
            os.makedirs(path, exist_ok=True)
            configuration = self.get_config()
            configuration["path"] = path
            torch.save(configuration, os.path.join(path, config))
            torch.save(self.state_dict(), os.path.join(path, weights))

    @classmethod
    def load(
        cls,
        path: str,
        config: str = "config.pk",
        weights: str = "weights.pk",
        _mode: int = _TRAIN,
    ) -> None:
        r"""
        Load model from files

        Parameters
        ----------
        path
            Path to a model directory to load from
        config
            Name of the configuration file
        weights
            Name of the weights file
        """
        assert os.path.exists(path)

        configuration = torch.load(os.path.join(path, config))
        if configuration["_mode"] == _TEST and _mode == _TRAIN:
            raise RuntimeError(
                "The model was minimal, please use argument '_mode=Cell_BLAST.blast.MINIMAL'"
            )

        model = cls.load_config(configuration)
        model.load_state_dict(torch.load(os.path.join(path, weights), map_location=DEVICE), strict=False)

        return model

    def inference(
        self,
        adata: ad.AnnData,
        batch_size: int = 4096,
        n_posterior: int = 0,
        progress_bar: bool = False,
        priority: str = "auto",
        random_seed: typing.Optional[int] = config._USE_GLOBAL,
    ) -> np.ndarray:
        r"""
        Project expression profiles into the cell embedding space.

        Parameters
        ----------
        adata
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

        self.eval()
        self.to(DEVICE)

        random_seed = (
            config.RANDOM_SEED
            if random_seed is None or random_seed == config._USE_GLOBAL
            else random_seed
        )
        x = data.select_vars(adata, self.genes).X
        if "__libsize__" not in adata.obs.columns:
            data.compute_libsize(adata)
        l = adata.obs["__libsize__"].to_numpy().reshape((-1, 1))
        if n_posterior > 0:
            if priority == "auto":
                priority = "memory" if x.shape[0] > 1e4 else "speed"
            if priority == "speed":
                if scipy.sparse.issparse(x):
                    xrep = x.tocsr()[np.repeat(np.arange(x.shape[0]), n_posterior)]
                else:
                    xrep = np.repeat(x, n_posterior, axis=0)
                lrep = np.repeat(l, n_posterior, axis=0)
                data_dict = OrderedDict(exprs=xrep, library_size=lrep)
                return (
                    self._fetch_latent(
                        data.Dataset(data_dict),
                        batch_size,
                        True,
                        progress_bar,
                        random_seed,
                    )
                    .astype(np.float32)
                    .reshape((x.shape[0], n_posterior, -1))
                )
            else:  # priority == "memory":
                data_dict = OrderedDict(exprs=x, library_size=l)
                return np.stack(
                    [
                        self._fetch_latent(
                            data.Dataset(data_dict),
                            batch_size,
                            True,
                            progress_bar,
                            (random_seed + i) if random_seed is not None else None,
                        ).astype(np.float32)
                        for i in range(n_posterior)
                    ],
                    axis=1,
                )
        data_dict = OrderedDict(exprs=x, library_size=l)
        return self._fetch_latent(
            data.Dataset(data_dict), batch_size, False, progress_bar, random_seed
        ).astype(np.float32)

    def clustering(
        self,
        adata: ad.AnnData,
        batch_size: int = 4096,
        return_confidence: bool = False,
        progress_bar: bool = False,
    ) -> typing.Tuple[np.ndarray]:
        r"""
        Get model intrinsic clustering of the data.

        Parameters
        ----------
        adata
            Dataset for which to obtain the intrinsic clustering.
        batch_size
            Minibatch size.
            Changing this may slighly affect speed, but not the result.
        return_confidence
            Whether to return model intrinsic clustering confidence.
        progress_bar
            Whether to show progress bar during projection.

        Returns
        -------
        idx
            model intrinsic clustering index, 1 dimensional
        confidence (if ``return_confidence`` is True)
            model intrinsic clustering confidence, 1 dimensional
        """

        self.eval()
        self.to(DEVICE)

        if not isinstance(self.latent_module, latent.CatGau):
            raise Exception("Model has no intrinsic clustering")
        x = data.select_vars(adata, self.genes).X
        if "__libsize__" not in adata.obs.columns:
            data.compute_libsize(adata)
        l = adata.obs["__libsize__"].to_numpy().reshape((-1, 1))
        data_dict = OrderedDict(exprs=x, library_size=l)
        cat = self._fetch_cat(
            data.Dataset(data_dict), batch_size, False, progress_bar
        ).astype(np.float32)
        if return_confidence:
            return cat.argmax(axis=1), cat.max(axis=1)
        return cat.argmax(axis=1)

    def gene_grad(
        self,
        adata: ad.AnnData,
        latent_grad: np.ndarray,
        batch_size: int = 4096,
        progress_bar: bool = False,
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

        self.eval()
        self.to(DEVICE)

        x = data.select_vars(adata, self.genes).X
        if "__libsize__" not in adata.obs.columns:
            data.compute_libsize(adata)
        l = adata.obs["__libsize__"].to_numpy().reshape((-1, 1))
        data_dict = OrderedDict(exprs=x, library_size=l, output_grad=latent_grad)
        return self._fetch_grad(
            data.Dataset(data_dict), batch_size=batch_size, progress_bar=progress_bar
        )

    def _fetch_latent(
        self,
        dataset: data.Dataset,
        batch_size: int,
        noisy: bool,
        progress_bar: bool,
        random_seed: int,
    ) -> np.ndarray:
        self.ensure_reproducibility(random_seed)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if progress_bar:
            dataloader = utils.smart_tqdm()(dataloader)

        with torch.no_grad():
            latents = []
            for feed_dict in dataloader:
                for key, value in feed_dict.items():
                    feed_dict[key] = value.to(DEVICE)
                exprs = feed_dict["exprs"]
                libs = feed_dict["library_size"]
                latents.append(
                    self.latent_module.fetch_latent(self.preprocess(exprs, libs, noisy))
                )
        return torch.cat(latents).cpu().numpy()

    def _fetch_cat(
        self, dataset: data.Dataset, batch_size: int, noisy: bool, progress_bar: bool
    ) -> typing.Tuple[np.ndarray]:
        self.ensure_reproducibility(self.random_seed)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if progress_bar:
            dataloader = utils.smart_tqdm()(dataloader)

        with torch.no_grad():
            cats = []
            for feed_dict in dataloader:
                for key, value in feed_dict.items():
                    feed_dict[key] = value.to(DEVICE)
                exprs = feed_dict["exprs"]
                libs = feed_dict["library_size"]
                cats.append(
                    self.latent_module.fetch_cat(self.preprocess(exprs, libs, noisy))
                )
        return torch.cat(cats).cpu().numpy()

    def _fetch_grad(
        self, dataset: data.Dataset, batch_size: int, progress_bar: bool
    ) -> np.ndarray:
        self.ensure_reproducibility(self.random_seed)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if progress_bar:
            dataloader = utils.smart_tqdm()(dataloader)

        grads = []
        for feed_dict in dataloader:
            for key, value in feed_dict.items():
                feed_dict[key] = value.to(DEVICE)
            exprs = feed_dict["exprs"]
            libs = feed_dict["library_size"]
            latent_grad = feed_dict["output_grad"]
            grads.append(
                self.latent_module.fetch_grad(
                    self.preprocess(exprs, libs, self.denoising), latent_grad
                )
            )
        return torch.cat(grads).cpu().numpy()


def fit_DIRECTi(
    adata: ad.AnnData,
    genes: typing.Optional[typing.List[str]] = None,
    supervision: typing.Optional[str] = None,
    batch_effect: typing.Optional[typing.List[str]] = None,
    latent_dim: int = 10,
    cat_dim: typing.Optional[int] = None,
    h_dim: int = 128,
    depth: int = 1,
    prob_module: str = "NB",
    rmbatch_module: typing.Union[str, typing.List[str]] = "Adversarial",
    latent_module_kwargs: typing.Optional[typing.Mapping] = None,
    prob_module_kwargs: typing.Optional[typing.Mapping] = None,
    rmbatch_module_kwargs: typing.Optional[
        typing.Union[typing.Mapping, typing.List[typing.Mapping]]
    ] = None,
    optimizer: str = "RMSPropOptimizer",
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    val_split: float = 0.1,
    epoch: int = 1000,
    patience: int = 30,
    progress_bar: bool = False,
    reuse_weights: typing.Optional[str] = None,
    random_seed: int = config._USE_GLOBAL,
    path: typing.Optional[str] = None,
) -> DIRECTi:
    r"""
    A convenient one-step function to build and fit DIRECTi models.
    Should work well in most cases.

    Parameters
    ----------
    adata
        Dataset to be fitted.
    genes
        Genes to fit on, should be a subset of :attr:`anndata.AnnData.var_names`.
        If not specified, all genes are used.
    supervision
        Specifies a column in the :attr:`anndata.AnnData.obs` table for use as
        (semi-)supervision. If value in the specified column is emtpy,
        the corresponding cells will be treated as unsupervised.
    batch_effect
        Specifies one or more columns in the :attr:`anndata.AnnData.obs` table
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

    random_seed = (
        config.RANDOM_SEED
        if random_seed is None or random_seed == config._USE_GLOBAL
        else random_seed
    )
    DIRECTi.ensure_reproducibility(random_seed)

    if latent_module_kwargs is None:
        latent_module_kwargs = {}
    if prob_module_kwargs is None:
        prob_module_kwargs = {}
    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}

    if genes is None:
        genes = adata.var_names.values
    if isinstance(genes, (pd.Series, pd.Index)):
        genes = genes.to_numpy()
    if isinstance(genes, np.ndarray):
        genes = genes.tolist()
    assert isinstance(genes, list)
    if "__libsize__" not in adata.obs.columns:
        data.compute_libsize(adata)
    data_dict = OrderedDict(
        library_size=adata.obs["__libsize__"].to_numpy().reshape((-1, 1)),
        exprs=data.select_vars(adata, genes).X,
    )

    if batch_effect is None:
        batch_effect = []
    elif isinstance(batch_effect, str):
        batch_effect = [batch_effect]
    elif isinstance(batch_effect, pd.Series):
        batch_effect = batch_effect.values
    elif isinstance(batch_effect, np.ndarray):
        batch_effect = batch_effect.tolist()
    assert isinstance(batch_effect, list)
    for _batch_effect in batch_effect:
        data_dict[_batch_effect] = utils.encode_onehot(
            adata.obs[_batch_effect], sort=True
        )  # sorting ensures batch order reproducibility for later tuning
    if supervision is not None:
        data_dict[supervision] = utils.encode_onehot(
            adata.obs[supervision], sort=True
        )  # sorting ensures supervision order reproducibility for later tuning
        if cat_dim is None:
            cat_dim = data_dict[supervision].shape[1]
        elif cat_dim > data_dict[supervision].shape[1]:
            data_dict[supervision] = scipy.sparse.hstack(
                [
                    data_dict[supervision].tocsc(),
                    scipy.sparse.csc_matrix(
                        (
                            data_dict[supervision].shape[0],
                            cat_dim - data_dict[supervision].shape[1],
                        )
                    ),
                ]
            ).tocsr()
        elif cat_dim < data_dict[supervision].shape[1]:  # pragma: no cover
            raise ValueError(
                "`cat_dim` must be greater than or equal to "
                "number of supervised classes!"
            )
        # else ==

    kwargs = dict(input_dim=len(genes), latent_dim=latent_dim, h_dim=h_dim, depth=depth)
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

    rmbatch_list = nn.ModuleList()
    full_latent_dim = [latent_dim]
    for _batch_effect, _rmbatch_module, _rmbatch_module_kwargs in zip(
        batch_effect, rmbatch_module, rmbatch_module_kwargs
    ):
        batch_dim = len(adata.obs[_batch_effect].dropna().unique())
        full_latent_dim.append(batch_dim)
        kwargs = dict(batch_dim=batch_dim, latent_dim=latent_dim, name=_batch_effect)
        if _rmbatch_module in (
            "Adversarial",
            "MNNAdversarial",
            "AdaptiveMNNAdversarial",
        ):
            kwargs.update(dict(h_dim=h_dim, depth=depth))
            kwargs.update(_rmbatch_module_kwargs)
        elif _rmbatch_module not in ("RMBatch", "MNN"):  # pragma: no cover
            raise ValueError("Invalid rmbatch method!")
        # else "RMBatch" or "MNN"
        kwargs.update(_rmbatch_module_kwargs)
        rmbatch_list.append(getattr(rmbatch, _rmbatch_module)(**kwargs))

    kwargs = dict(
        output_dim=len(genes), full_latent_dim=full_latent_dim, h_dim=h_dim, depth=depth
    )
    kwargs.update(prob_module_kwargs)
    prob_module = getattr(prob, prob_module)(**kwargs)

    model = DIRECTi(
        genes=genes,
        latent_module=latent_module,
        prob_module=prob_module,
        rmbatch_modules=rmbatch_list,
        learning_rate=learning_rate,
        path=path,
        random_seed=random_seed,
    )

    if not reuse_weights is None:
        model.load_state_dict(torch.load(reuse_weights, map_location=DEVICE))

    if optimizer != "RMSPropOptimizer":
        utils.logger.warning("Argument `optimizer` is no longer supported!")

    model.fit(
        dataset=data.Dataset(data_dict),
        batch_size=batch_size,
        val_split=val_split,
        epoch=epoch,
        patience=patience,
        progress_bar=progress_bar,
    )

    return model


def align_DIRECTi(
    model: DIRECTi,
    original_adata: ad.AnnData,
    new_adata: typing.Union[ad.AnnData, typing.Mapping[str, ad.AnnData]],
    rmbatch_module: str = "MNNAdversarial",
    rmbatch_module_kwargs: typing.Optional[typing.Mapping] = None,
    deviation_reg: float = 0.01,
    optimizer: str = "RMSPropOptimizer",
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    val_split: float = 0.1,
    epoch: int = 100,
    patience: int = 100,
    tolerance: float = 0.0,
    reuse_weights: bool = True,
    progress_bar: bool = False,
    random_seed: int = config._USE_GLOBAL,
    path: typing.Optional[str] = None,
) -> DIRECTi:
    r"""
    Align datasets starting with an existing DIRECTi model (fine-tuning)

    Parameters
    ----------
    model
        A pretrained DIRECTi model.
    original_adata
        The dataset that the model was originally trained on.
    new_adata
        A new dataset or a dictionary containing new datasets,
        to be aligned with ``original_dataset``.
    rmbatch_module
        Specifies the batch effect correction method to use for aligning new
        datasets.
    rmbatch_module_kwargs
        Keyword arguments to be passed to the rmbatch module.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    optimizer
        Name of optimizer used in training.
    learning_rate
        Learning rate used in training.
    batch_size
        Size of minibatches used in training.
    val_split
        Fraction of data to use for validation.
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

    random_seed = (
        config.RANDOM_SEED if random_seed == config._USE_GLOBAL else random_seed
    )
    DIRECTi.ensure_reproducibility(random_seed)

    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}
    if rmbatch_module_kwargs is None:
        rmbatch_module_kwargs = {}
    if isinstance(new_adata, ad.AnnData):
        new_adatas = {"__new__": new_adata}
    elif isinstance(new_adata, dict):
        assert (
            "__original__" not in new_adata
        ), "Key `__original__` is now allowed in new datasets."
        new_adatas = new_adata.copy()  # shallow
    else:
        raise TypeError("Invalid type for argument `new_dataset`.")

    _config = model.get_config()
    for _rmbatch_module in _config["rmbatch_modules"]:
        _rmbatch_module["delay"] = 0
    kwargs = {
        "batch_dim": len(new_adatas) + 1,
        "latent_dim": model.latent_module.latent_dim,
        "delay": 0,
        "name": "__align__",
        "_class": rmbatch_module,
    }
    if rmbatch_module in ("Adversarial", "MNNAdversarial", "AdaptiveMNNAdversarial"):
        kwargs.update(
            dict(
                h_dim=model.latent_module.h_dim,
                depth=model.latent_module.depth,
                dropout=model.latent_module.dropout,
                lambda_reg=0.01,
            )
        )
    elif rmbatch_module not in ("RMBatch", "MNN"):  # pragma: no cover
        raise ValueError("Unknown rmbatch_module!")
    # else "RMBatch" or "MNN"
    kwargs.update(rmbatch_module_kwargs)
    _config["rmbatch_modules"].append(kwargs)

    _config["prob_module"]["full_latent_dim"].append(len(new_adatas) + 1)
    _config["prob_module"]["fine_tune"] = True
    _config["prob_module"]["deviation_reg"] = deviation_reg
    _config["learning_rate"] = learning_rate
    _config["path"] = path

    aligned_model = DIRECTi.load_config(_config)
    if reuse_weights:
        aligned_model.load_state_dict(model.state_dict(), strict=False)
    supervision = (
        aligned_model.latent_module.name
        if isinstance(aligned_model.latent_module, latent.SemiSupervisedCatGau)
        else None
    )

    assert (
        "__align__" not in original_adata.obs.columns
    ), "Please remove column `__align__` from obs of the original dataset."
    original_adata = ad.AnnData(
        X=original_adata.X,
        obs=original_adata.obs.copy(deep=False),
        var=original_adata.var.copy(deep=False),
    )
    if "__libsize__" not in original_adata.obs.columns:
        data.compute_libsize(original_adata)
    original_adata = data.select_vars(original_adata, model.genes)
    for key in new_adatas.keys():
        assert (
            "__align__" not in new_adatas[key].obs.columns
        ), f"Please remove column `__align__` from new dataset {key}."
        new_adatas[key] = ad.AnnData(
            X=new_adatas[key].X,
            obs=new_adatas[key].obs.copy(deep=False),
            var=new_adatas[key].var.copy(deep=False),
        )
        new_adatas[key].obs = new_adatas[key].obs.loc[
            :, new_adatas[key].obs.columns == "__libsize__"
        ]  # All meta in new datasets are cleared to avoid interference
        if "__libsize__" not in new_adatas[key].obs.columns:
            data.compute_libsize(new_adatas[key])
        new_adatas[key] = data.select_vars(new_adatas[key], model.genes)

    adatas = {"__original__": original_adata, **new_adatas}
    for key, val in adatas.items():
        val.obs["__align__"] = key
    adata = ad.concat(adatas, join="outer", fill_value=0)

    data_dict = OrderedDict(
        library_size=adata.obs["__libsize__"].to_numpy().reshape((-1, 1)),
        exprs=data.select_vars(adata, model.genes).X,  # Ensure order
    )
    for rmbatch_module in aligned_model.rmbatch_modules:
        data_dict[rmbatch_module.name] = utils.encode_onehot(
            adata.obs[rmbatch_module.name], sort=True
        )
    if isinstance(aligned_model.latent_module, latent.SemiSupervisedCatGau):
        data_dict[supervision] = utils.encode_onehot(adata.obs[supervision], sort=True)
        cat_dim = aligned_model.latent_module.cat_dim
        if cat_dim > data_dict[supervision].shape[1]:
            data_dict[supervision] = scipy.sparse.hstack(
                [
                    data_dict[supervision].tocsc(),
                    scipy.sparse.csc_matrix(
                        (
                            data_dict[supervision].shape[0],
                            cat_dim - data_dict[supervision].shape[1],
                        )
                    ),
                ]
            ).tocsr()

    if optimizer != "RMSPropOptimizer":
        utils.logger.warning("Argument `optimizer` is not supported!")

    aligned_model.fit(
        dataset=data.Dataset(data_dict),
        batch_size=batch_size,
        val_split=val_split,
        epoch=epoch,
        patience=patience,
        tolerance=tolerance,
        progress_bar=progress_bar,
    )
    return aligned_model
