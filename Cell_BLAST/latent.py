r"""
Latent space / encoder modules for DIRECTi
"""

import itertools
import typing

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn

from . import config, utils
from .rebuild import MLP, Linear


class Regularizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        name: str = "Reg",
        _class: str = "Regularizer",
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.name = name
        self._class = _class

        for key in kwargs.keys():
            utils.logger.warning("Argument `%s` is no longer supported!" % key)

        i_dim = [latent_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        if depth > 0:
            dropout[0] = 0.0
        self.mlp = MLP(i_dim, o_dim, dropout)
        self.output = Linear(h_dim, 1) if depth > 0 else Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.output(self.mlp(x)))

    def get_config(self) -> typing.Mapping:
        return {
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "name": self.name,
            "_class": self._class,
        }


class Latent(nn.Module):
    r"""
    Abstract base class for latent variable modules.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "Latent",
        _class: str = "Latent",
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.fine_tune = fine_tune
        self.deviation_reg = deviation_reg
        self.name = name
        self._class = _class
        self.record_prefix = "discriminator"

        for key in kwargs.keys():
            utils.logger.warning("Argument `%s` is no longer supported!" % key)

    @staticmethod
    def gan_d_loss(
        y: torch.Tensor, y_hat: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        return -(torch.log(y_hat + eps) + torch.log(1 - y + eps)).mean()

    @staticmethod
    def gan_g_loss(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -torch.log(y + eps).mean()

    def get_config(self) -> typing.Mapping:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "fine_tune": self.fine_tune,
            "deviation_reg": self.deviation_reg,
            "name": self.name,
            "_class": self._class,
        }


class Gau(Latent):
    r"""
    Build a Gaussian latent module. The Gaussian latent variable is used as
    cell embedding.

    Parameters
    ----------
    input_dim
        Dimensionality of the input tensor.
    latent_dim
        Dimensionality of the latent variable.
    h_dim
        Dimensionality of the hidden layers in the encoder MLP.
    depth
        Number of hidden layers in the encoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        Regularization strength on the latent variable.
    name
        Name of the module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.001,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "Gau",
        _class: str = "Gau",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            latent_dim,
            h_dim,
            depth,
            dropout,
            lambda_reg,
            fine_tune,
            deviation_reg,
            name,
            _class,
            **kwargs,
        )

        self.gau_reg = Regularizer(latent_dim, h_dim, depth, dropout, name="gau")
        self.gaup_sampler = D.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))

        i_dim = [input_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        self.mlp = MLP(i_dim, o_dim, dropout, bias=False, batch_normalization=True)
        self.gau = (
            Linear(h_dim, latent_dim) if depth > 0 else Linear(input_dim, latent_dim)
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mlp.first_layer_trainable = False
        self.gau.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss() + self.gau.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gau = self.gau(self.mlp(x))
        return gau, gau

    def fetch_latent(self, x: torch.Tensor) -> torch.Tensor:
        gau = self.gau(self.mlp(x))
        return gau

    def fetch_cat(self, x: torch.Tensor) -> torch.Tensor:
        raise Exception("Model has no intrinsic clustering")

    def fetch_grad(self, x: torch.Tensor, latent_grad: torch.Tensor) -> torch.Tensor:
        x_with_grad = x.requires_grad_(True)

        gau = self.gau(self.mlp(x))
        gau.backward(latent_grad)

        return x_with_grad.grad

    def d_loss(
        self, gau: torch.Tensor, feed_dict: typing.Mapping, loss_record: typing.Mapping
    ) -> typing.Tuple[torch.Tensor]:
        gaup = self.gaup_sampler.sample((gau.shape[0], self.latent_dim)).to(
            config.DEVICE
        )
        gau_pred = self.gau_reg(gau)
        gaup_pred = self.gau_reg(gaup)
        gau_d_loss = self.gan_d_loss(gau_pred, gaup_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] += (gau_d_loss.item() * gau.shape[0])

        return self.lambda_reg * gau_d_loss

    def g_loss(
        self, gau: torch.Tensor, feed_dict: typing.Mapping, loss_record: typing.Mapping
    ) -> typing.Tuple[torch.Tensor]:
        gau_pred = self.gau_reg(gau)
        gau_g_loss = self.gan_g_loss(gau_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] += (gau_g_loss.item() * gau.shape[0])

        if self.fine_tune:
            return (
                self.lambda_reg * gau_g_loss
                + self.deviation_reg * self.deviation_loss()
            )
        else:
            return self.lambda_reg * gau_g_loss

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] = 0

    def parameters_reg(self):
        return self.gau_reg.parameters()

    def parameters_fit(self):
        return itertools.chain(
            self.mlp.parameters(),
            self.gau.parameters(),
        )

    def get_config(self) -> typing.Mapping:
        return {**super().get_config()}


class CatGau(Latent):
    r"""
    Build a double latent module, with a continuous Gaussian latent variable
    and a one-hot categorical latent variable for intrinsic clustering of
    the data. These two latent variabels are then combined into a single
    cell embedding vector.

    Parameters
    ----------
    input_dim
        Dimensionality of the input tensor.
    latent_dim
        Dimensionality of the latent variable.
    cat_dim
        Number of intrinsic clusters.
    h_dim
        Dimensionality of the hidden layers in the encoder MLP.
    depth
        Number of hidden layers in the encoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        Regularization strength on the latent variable.
    name
        Name of the module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        cat_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.001,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "CatGau",
        _class: str = "CatGau",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            latent_dim,
            h_dim,
            depth,
            dropout,
            lambda_reg,
            fine_tune,
            deviation_reg,
            name,
            _class,
            **kwargs,
        )
        self.cat_dim = cat_dim

        self.gau_reg = Regularizer(latent_dim, h_dim, depth, dropout, name="gau")
        self.gaup_sampler = D.Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
        self.cat_reg = Regularizer(cat_dim, h_dim, depth, dropout, name="cat")
        self.catp_sampler = D.OneHotCategorical(probs=torch.ones(cat_dim) / cat_dim)

        i_dim = [input_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        self.mlp = MLP(i_dim, o_dim, dropout, bias=False, batch_normalization=True)
        self.gau = (
            Linear(h_dim, latent_dim) if depth > 0 else Linear(input_dim, latent_dim)
        )
        self.cat = Linear(h_dim, cat_dim) if depth > 0 else Linear(input_dim, cat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.mat = Linear(cat_dim, latent_dim, bias=False, init_std=0.1, trunc=False)

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mlp.first_layer_trainable = False
        self.gau.save_origin_state()
        self.cat.save_origin_state()
        self.mat.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss()
            + self.gau.deviation_loss()
            + self.cat.deviation_loss()
            + self.mat.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        x = self.mlp(x)
        gau = self.gau(x)
        cat = self.softmax(self.cat(x))
        latent = gau + self.mat(cat)
        return latent, (gau, cat)

    def fetch_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        gau = self.gau(x)
        cat = self.softmax(self.cat(x))
        latent = gau + self.mat(cat)
        return latent

    def fetch_cat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        cat = self.softmax(self.cat(x))
        return cat

    def fetch_grad(self, x: torch.Tensor, latent_grad: torch.Tensor) -> torch.Tensor:
        x_with_grad = x.requires_grad_(True)

        x = self.mlp(x)
        gau = self.gau(x)
        cat = self.softmax(self.cat(x))
        latent = gau + self.mat(cat)
        latent.backward(latent_grad)

        return x_with_grad.grad

    def d_loss(
        self,
        catgau: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        gau, cat = catgau

        gaup = self.gaup_sampler.sample((gau.shape[0], self.latent_dim)).to(
            config.DEVICE
        )
        gau_pred = self.gau_reg(gau)
        gaup_pred = self.gau_reg(gaup)
        gau_d_loss = self.gan_d_loss(gau_pred, gaup_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] += (gau_d_loss.item() * gau.shape[0])

        catp = self.catp_sampler.sample((cat.shape[0],)).to(config.DEVICE)
        cat_pred = self.cat_reg(cat)
        catp_pred = self.cat_reg(catp)
        cat_d_loss = self.gan_d_loss(cat_pred, catp_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/d_loss/d_loss"
        ] += (cat_d_loss.item() * cat.shape[0])

        return self.lambda_reg * (gau_d_loss + cat_d_loss)

    def g_loss(
        self,
        catgau: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> typing.Tuple[torch.Tensor]:
        gau, cat = catgau

        gau_pred = self.gau_reg(gau)
        gau_g_loss = self.gan_g_loss(gau_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] += (gau_g_loss.item() * gau.shape[0])

        cat_pred = self.cat_reg(cat)
        cat_g_loss = self.gan_g_loss(cat_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/g_loss/g_loss"
        ] += (cat_g_loss.item() * cat.shape[0])

        if self.fine_tune:
            return (
                self.lambda_reg * (gau_g_loss + cat_g_loss)
                + self.deviation_reg * self.deviation_loss()
            )
        else:
            return self.lambda_reg * (gau_g_loss + cat_g_loss)

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/d_loss/d_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/g_loss/g_loss"
        ] = 0

    def parameters_reg(self):
        return itertools.chain(self.gau_reg.parameters(), self.cat_reg.parameters())

    def parameters_fit(self):
        return itertools.chain(
            self.mlp.parameters(),
            self.gau.parameters(),
            self.cat.parameters(),
            self.mat.parameters(),
        )

    def get_config(self) -> typing.Mapping:
        return {"cat_dim": self.cat_dim, **super().get_config()}


class SemiSupervisedCatGau(CatGau):
    r"""
    Build a double latent module, with a continuous Gaussian latent variable
    and a one-hot categorical latent variable for intrinsic clustering of
    the data. The categorical latent supports semi-supervision. The two latent
    variables are then combined into a single cell embedding vector.

    Parameters
    ----------
    input_dim
        Dimensionality of the input tensor.
    latent_dim
        Dimensionality of the Gaussian latent variable.
    cat_dim
        Number of intrinsic clusters.
    h_dim
        Dimensionality of the hidden layers in the encoder MLP.
    depth
        Number of hidden layers in the encoder MLP.
    dropout
        Dropout rate.
    lambda_sup
        Supervision strength.
    background_catp
        Unnormalized background prior distribution of the intrinsic
        clustering latent.
        For each supervised cell in a minibatch, unnormalized prior
        probability of the corresponding cluster will increase by 1,
        so this parameter determines how much to trust supervision class
        frequency, and it balances between supervision and identifying new
        clusters.
    lambda_reg
        Regularization strength on the latent variables.
    name
        Name of latent module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        cat_dim: int,
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_sup: float = 10.0,
        background_catp: float = 1e-3,
        lambda_reg: float = 0.001,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "SemiSupervisedCatGau",
        _class: str = "SemiSupervisedCatGau",
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            latent_dim,
            cat_dim,
            h_dim,
            depth,
            dropout,
            lambda_reg,
            fine_tune,
            deviation_reg,
            name,
            _class,
            **kwargs,
        )
        self.lambda_sup = lambda_sup
        self.background_catp = background_catp

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        x = self.mlp(x)
        gau = self.gau(x)
        cat_logit = self.cat(x)
        cat = self.softmax(cat_logit)
        latent = gau + self.mat(cat)
        return latent, (gau, cat, cat_logit)

    def d_loss(
        self,
        catgau: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        gau, cat, _ = catgau
        cats = feed_dict[self.name]

        gaup = self.gaup_sampler.sample((gau.shape[0], self.latent_dim)).to(
            config.DEVICE
        )
        gau_pred = self.gau_reg(gau)
        gaup_pred = self.gau_reg(gaup)
        gau_d_loss = self.gan_d_loss(gau_pred, gaup_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] += (gau_d_loss.item() * gau.shape[0])

        cat_prob = torch.ones(self.cat_dim) * self.background_catp + cats.cpu().sum(
            dim=0
        )
        catp_sampler = D.OneHotCategorical(probs=cat_prob / cat_prob.sum())
        catp = catp_sampler.sample((cat.shape[0],)).to(config.DEVICE)
        cat_pred = self.cat_reg(cat)
        catp_pred = self.cat_reg(catp)
        cat_d_loss = self.gan_d_loss(cat_pred, catp_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/d_loss/d_loss"
        ] += (cat_d_loss.item() * cat.shape[0])

        return self.lambda_reg * (gau_d_loss + cat_d_loss)

    def g_loss(
        self,
        catgau: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> typing.Tuple[torch.Tensor]:
        gau, cat, cat_logit = catgau
        cats = feed_dict[self.name]

        mask = cat.sum(dim=1) > 0
        if mask.sum() > 0:
            sup_loss = F.cross_entropy(cat_logit[mask], cats[mask].argmax(dim=1))
        else:
            sup_loss = torch.tensor(0)
        loss_record["semi_supervision/" + self.name + "/supervised_loss"] += (
            sup_loss.item() * cats.shape[0]
        )

        gau_pred = self.gau_reg(gau)
        gau_g_loss = self.gan_g_loss(gau_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] += (gau_g_loss.item() * gau.shape[0])

        cat_pred = self.cat_reg(cat)
        cat_g_loss = self.gan_g_loss(cat_pred)
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/g_loss/g_loss"
        ] += (cat_g_loss.item() * cat.shape[0])

        if self.fine_tune:
            return (
                self.lambda_sup * sup_loss
                + self.lambda_reg * (gau_g_loss + cat_g_loss)
                + self.deviation_reg * self.deviation_loss()
            )
        else:
            return self.lambda_sup * sup_loss + self.lambda_reg * (
                gau_g_loss + cat_g_loss
            )

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/d_loss/d_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.gau_reg.name
            + "/g_loss/g_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/d_loss/d_loss"
        ] = 0
        loss_record[
            self.record_prefix
            + "/"
            + self.name
            + "/"
            + self.cat_reg.name
            + "/g_loss/g_loss"
        ] = 0
        loss_record["semi_supervision/" + self.name + "/supervised_loss"] = 0

    def get_config(self) -> typing.Mapping:
        return {
            "lambda_sup": self.lambda_sup,
            "background_catp": self.background_catp,
            **super().get_config(),
        }
