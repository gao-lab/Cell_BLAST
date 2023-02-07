r"""
Batch effect removing modules for DIRECTi
"""

import typing

import torch
import torch.nn.functional as F
from torch import nn

from . import config, utils
from .rebuild import MLP, Linear


class RMBatch(nn.Module):
    r"""
    Parent class for systematical bias / batch effect removal modules.
    """

    def __init__(
        self,
        batch_dim: int,
        latent_dim: int,
        delay: int = 20,
        name: str = "RMBatch",
        _class: str = "RMBatch",
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_dim = batch_dim
        self.latent_dim = latent_dim
        self.delay = delay
        self.name = name
        self._class = _class
        self.record_prefix = "discriminator"
        self.n_steps = 0

        for key in kwargs.keys():
            utils.logger.warning("Argument `%s` is no longer supported!" % key)

    def get_mask(self, x: torch.Tensor, feed_dict: typing.Mapping) -> torch.Tensor:
        b = feed_dict[self.name]
        return b.sum(dim=1) > 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x[mask]

    def d_loss(
        self,
        x: torch.Tensor,
        feed_dict: typing.Mapping,
        mask: torch.Tensor,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        return torch.tensor(0)

    def g_loss(
        self,
        x: torch.Tensor,
        feed_dict: typing.Mapping,
        mask: torch.Tensor,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        return torch.tensor(0)

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        pass

    def get_config(self) -> typing.Mapping:
        return {
            "batch_dim": self.batch_dim,
            "latent_dim": self.latent_dim,
            "delay": self.delay,
            "name": self.name,
            "_class": self._class,
        }


class Adversarial(RMBatch):
    r"""
    Build a batch effect correction module that uses adversarial batch alignment.

    Parameters
    ----------
    batch_dim
        Number of batches.
    latent_dim
        Dimensionality of the latent variable.
    h_dim
        Dimensionality of the hidden layers in the discriminator MLP.
    depth
        Number of hidden layers in the discriminator MLP.
    dropout
        Dropout rate.
    lambda_reg
        Strength of batch effect correction,
    n_steps
        How many discriminator steps to run for each encoder step.
    delay
        How many epoches to delay before using Adversarial batch correction.
    name
        Name of the module.
    """

    def __init__(
        self,
        batch_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.01,
        n_steps: int = 1,
        delay: int = 20,
        name: str = "AdvBatch",
        _class: str = "Adversarial",
        **kwargs,
    ) -> None:
        super().__init__(batch_dim, latent_dim, delay, name, _class, **kwargs)
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.n_steps = n_steps

        i_dim = [latent_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        if depth > 0:
            dropout[0] = 0.0
        self.mlp = MLP(i_dim, o_dim, dropout)
        self.pred = (
            Linear(h_dim, batch_dim) if depth > 0 else Linear(latent_dim, batch_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pred(self.mlp(x[mask]))

    def d_loss(
        self,
        pred: torch.Tensor,
        feed_dict: typing.Mapping,
        mask: torch.Tensor,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        b = feed_dict[self.name]
        rmbatch_d_loss = F.cross_entropy(pred, b[mask].argmax(dim=1))
        loss_record[self.record_prefix + "/" + self.name + "/d_loss"] += (
            rmbatch_d_loss.item() * b.shape[0]
        )

        return self.lambda_reg * rmbatch_d_loss

    def g_loss(
        self,
        pred: torch.Tensor,
        feed_dict: typing.Mapping,
        mask: torch.Tensor,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        b = feed_dict[self.name]
        rmbatch_g_loss = F.cross_entropy(pred, b[mask].argmax(dim=1))

        return -self.lambda_reg * rmbatch_g_loss

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[self.record_prefix + "/" + self.name + "/d_loss"] = 0

    def get_config(self) -> typing.Mapping:
        return {
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            **super().get_config(),
        }


class MNN(RMBatch):
    r"""
    Build a batch effect correction module that uses mutual nearest neighbor
    (MNN) distance regularization.

    Parameters
    ----------
    batch_dim
        Number of batches.
    n_neighbors
        Number of nearest neighbors to use when selecting mutual nearest
        neighbors.
    lambda_reg
        Strength of batch effect correction.
    delay
        How many epoches to delay before using MNN batch correction.
    name
        Name of the module.
    """

    def __init__(
        self,
        batch_dim: int,
        latent_dim: int,
        n_neighbors: int = 5,
        lambda_reg: float = 1.0,
        delay: int = 20,
        name: str = "MNN",
        _class: str = "MNN",
        **kwargs,
    ) -> None:
        super().__init__(batch_dim, latent_dim, delay, name, _class, **kwargs)
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg

    @staticmethod
    def _neighbor_mask(d: torch.Tensor, k: int) -> torch.Tensor:
        n = d.shape[1]
        _, idx = d.topk(min(k, n), largest=False)
        return F.one_hot(idx, n).sum(dim=1) > 0

    @staticmethod
    def _mnn_mask(d: torch.Tensor, k: int) -> torch.Tensor:
        return MNN._neighbor_mask(d, k) & MNN._neighbor_mask(d.T, k).T

    def g_loss(
        self,
        x: torch.Tensor,
        feed_dict: typing.Mapping,
        mask: torch.Tensor,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        b = feed_dict[self.name]
        barg = b[mask].argmax(dim=1)
        masked_x = x[mask]
        x_grouping = []
        for i in range(b.shape[1]):
            x_grouping.append(masked_x[barg == i])
        penalties = []
        for i in range(b.shape[1]):
            for j in range(i + 1, b.shape[1]):
                if x_grouping[i].shape[0] > 0 and x_grouping[j].shape[0] > 0:
                    u = x_grouping[i].unsqueeze(1)
                    v = x_grouping[j].unsqueeze(0)
                    uv_dist = ((u - v).square()).sum(dim=2)
                    mnn_idx = self._mnn_mask(uv_dist, self.n_neighbors)
                    penalty = mnn_idx.float() * uv_dist
                    penalties.append(penalty.reshape(-1))
        penalties = torch.cat(penalties, dim=0)

        return self.lambda_reg * penalties.mean()

    def get_config(self) -> typing.Mapping:
        return {
            "n_neighbors": self.n_neighbors,
            "lambda_reg": self.lambda_reg,
            **super().get_config(),
        }


class MNNAdversarial(Adversarial):
    r"""
    Build a batch effect correction module that uses adversarial batch alignment
    among cells with mutual nearest neighbors.

    Parameters
    ----------
    batch_dim
        Number of batches.
    latent_dim
        Dimensionality of the latent variable.
    h_dim
        Dimensionality of the hidden layers in the discriminator MLP.
    depth
        Number of hidden layers in the discriminator MLP.
    dropout
        Dropout rate.
    lambda_reg
        Strength of batch effect correction,
    n_steps
        How many discriminator steps to run for each encoder step.
    n_neighbors
        Number of nearest neighbors to use when selecting mutual nearest
        neighbors.
    delay
        How many epoches to delay before using MNNAdversarial batch correction.
    name
        Name of the module.
    """

    def __init__(
        self,
        batch_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.01,
        n_steps: int = 1,
        n_neighbors: int = 5,
        delay: int = 20,
        name: str = "MNNAdvBatch",
        _class: str = "MNNAdversarial",
        **kwargs,
    ) -> None:
        super().__init__(
            batch_dim,
            latent_dim,
            h_dim,
            depth,
            dropout,
            lambda_reg,
            n_steps,
            delay,
            name,
            _class,
            **kwargs,
        )
        self.n_neighbors = n_neighbors

    @staticmethod
    def _neighbor_mask(d: torch.Tensor, k: int) -> torch.Tensor:
        n = d.shape[1]
        _, idx = d.topk(min(k, n), largest=False)
        return F.one_hot(idx, n).sum(dim=1) > 0

    @staticmethod
    def _mnn_mask(d: torch.Tensor, k: int) -> torch.Tensor:
        return (
            MNNAdversarial._neighbor_mask(d, k)
            & MNNAdversarial._neighbor_mask(d.T, k).T
        )

    def get_mask(self, x: torch.Tensor, feed_dict: typing.Mapping) -> torch.Tensor:
        b = feed_dict[self.name]
        mask = b.sum(dim=1) > 0
        mnn_mask = torch.zeros(b.shape[0], device=config.DEVICE) > 0
        masked_mnn_mask = mnn_mask[mask]
        barg = b[mask].argmax(dim=1)
        x_grouping = []
        for i in range(b.shape[1]):
            x_grouping.append(x[mask][barg == i].detach())
        for i in range(b.shape[1]):
            for j in range(i + 1, b.shape[1]):
                if x_grouping[i].shape[0] > 0 and x_grouping[j].shape[0] > 0:
                    u = x_grouping[i].unsqueeze(1)
                    v = x_grouping[j].unsqueeze(0)
                    uv_dist = ((u - v).square()).sum(dim=2)
                    mnn_idx = self._mnn_mask(uv_dist, self.n_neighbors)
                    masked_mnn_mask[barg == i] |= mnn_idx.sum(dim=1) > 0
                    masked_mnn_mask[barg == j] |= mnn_idx.sum(dim=0) > 0
        mnn_mask[mask] = masked_mnn_mask
        return mnn_mask

    def get_config(self) -> typing.Mapping:
        return {"n_neighbors": self.n_neighbors, **super().get_config()}


class AdaptiveMNNAdversarial(MNNAdversarial):
    def __init__(self, *args, **kwargs):
        utils.logger.warning(
            "RMBatch module `AdaptiveMNNAdversarial` is no longer supported, running as `MNNAdversarial`"
        )
        super().__init__(*args, **kwargs)
