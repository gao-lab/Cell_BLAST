r"""
Probabilistic / decoder modules for DIRECTi
"""

import math
import typing

import torch
import torch.nn.functional as F
from torch import nn

from . import utils
from .rebuild import MLP, Linear


class ProbModel(nn.Module):
    r"""
    Abstract base class for generative model modules.
    """

    def __init__(
        self,
        output_dim: int,
        full_latent_dim: typing.Tuple[int],
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "ProbModel",
        _class: str = "ProbModel",
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.full_latent_dim = full_latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.fine_tune = fine_tune
        self.deviation_reg = deviation_reg
        self.name = name
        self._class = _class
        self.record_prefix = "decoder"

        for key in kwargs.keys():
            utils.logger.warning("Argument `%s` is no longer supported!" % key)

        i_dim = [full_latent_dim] + [h_dim] * (depth - 1) if depth > 0 else []
        o_dim = [h_dim] * depth
        dropout = [dropout] * depth
        if depth > 0:
            dropout[0] = 0.0
        self.mlp = MLP(i_dim, o_dim, dropout)

    def get_config(self) -> typing.Mapping:
        return {
            "output_dim": self.output_dim,
            "full_latent_dim": self.full_latent_dim,
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "fine_tune": self.fine_tune,
            "deviation_reg": self.deviation_reg,
            "name": self.name,
            "_class": self._class,
        }


class NB(ProbModel):  # Negative binomial
    r"""
    Build a Negative Binomial generative module.

    Parameters
    ----------
    output_dim
        Dimensionality of the output tensor.
    full_latent_dim
        Dimensionality of the latent variable and Numbers of batches.
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        Regularization strength for the generative model parameters.
        Here log-scale variance of the scale parameter
        is regularized to improve numerical stability.
    fine_tune
        Whether the module is used in fine-tuning.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """

    def __init__(
        self,
        output_dim: int,
        full_latent_dim: typing.Tuple[int],
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "NB",
        _class: str = "NB",
        **kwargs,
    ) -> None:
        super().__init__(
            output_dim,
            full_latent_dim,
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

        self.mu = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )
        self.softmax = nn.Softmax(dim=1)
        self.log_theta = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mu.save_origin_state()
        self.log_theta.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss()
            + self.mu.deviation_loss()
            + self.log_theta.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    @staticmethod
    def log_likelihood(
        x: torch.Tensor, mu: torch.Tensor, log_theta: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        theta = torch.exp(log_theta)
        return (
            theta * log_theta
            - theta * torch.log(theta + mu + eps)
            + x * torch.log(mu + eps)
            - x * torch.log(theta + mu + eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )

    def forward(
        self, full_x: typing.Tuple[torch.Tensor], feed_dict: typing.Mapping
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        x = self.mlp(full_x)

        softmax_mu = self.softmax(self.mu(x))
        mu = softmax_mu * y.sum(dim=1, keepdim=True)
        log_theta = self.log_theta(x)
        return mu, log_theta

    def loss(
        self,
        mu_theta: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        mu, log_theta = mu_theta

        raw_loss = -self.log_likelihood(y, mu, log_theta).mean()
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] += (
            raw_loss.item() * mu.shape[0]
        )
        reg_loss = raw_loss + self.lambda_reg * log_theta.var()
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] += (
            reg_loss.item() * mu.shape[0]
        )

        if self.fine_tune:
            return reg_loss + self.deviation_reg * self.deviation_loss()
        else:
            return reg_loss

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] = 0
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] = 0


class ZINB(NB):
    r"""
    Build a Zero-Inflated Negative Binomial generative module.

    Parameters
    ----------
    output_dim
        Dimensionality of the output tensor.
    full_latent_dim
        Dimensionality of the latent variable and Numbers of batches.
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        Regularization strength for the generative model parameters.
        Here log-scale variance of the scale parameter
        is regularized to improve numerical stability.
    fine_tune
        Whether the module is used in fine-tuning.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """

    def __init__(
        self,
        output_dim: int,
        full_latent_dim: typing.Tuple[int],
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "ZINB",
        _class: str = "ZINB",
        **kwargs,
    ) -> None:
        super().__init__(
            output_dim,
            full_latent_dim,
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

        self.pi = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mu.save_origin_state()
        self.log_theta.save_origin_state()
        self.pi.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss()
            + self.mu.deviation_loss()
            + self.log_theta.deviation_loss()
            + self.pi.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    @staticmethod
    def log_likelihood(
        x: torch.Tensor,
        mu: torch.Tensor,
        log_theta: torch.Tensor,
        pi: torch.tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        theta = torch.exp(log_theta)
        case_zero = F.softplus(
            -pi + theta * log_theta - theta * torch.log(theta + mu + eps)
        ) - F.softplus(-pi)
        case_non_zero = (
            -pi
            - F.softplus(-pi)
            + theta * log_theta
            - theta * torch.log(theta + mu + eps)
            + x * torch.log(mu + eps)
            - x * torch.log(theta + mu + eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mask = (x < eps).float()
        res = mask * case_zero + (1 - mask) * case_non_zero
        return res

    def forward(
        self, full_x: typing.Tuple[torch.Tensor], feed_dict: typing.Mapping
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        x = self.mlp(full_x)

        softmax_mu = self.softmax(self.mu(x))
        mu = softmax_mu * y.sum(dim=1, keepdim=True)
        log_theta = self.log_theta(x)

        pi = self.pi(x)
        return mu, log_theta, pi

    def loss(
        self,
        mu_theta_pi: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        mu, log_theta, pi = mu_theta_pi

        raw_loss = -self.log_likelihood(y, mu, log_theta, pi).mean()
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] += (
            raw_loss.item() * mu.shape[0]
        )
        reg_loss = raw_loss + self.lambda_reg * log_theta.var()
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] += (
            reg_loss.item() * mu.shape[0]
        )

        if self.fine_tune:
            return reg_loss + self.deviation_reg * self.deviation_loss()
        else:
            return reg_loss


class LN(ProbModel):
    r"""
    Build a Log Normal generative module.

    Parameters
    ----------
    output_dim
        Dimensionality of the output tensor.
    full_latent_dim
        Dimensionality of the latent variable and Numbers of batches.
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        NOT USED.
    fine_tune
        Whether the module is used in fine-tuning.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """

    def __init__(
        self,
        output_dim: int,
        full_latent_dim: typing.Tuple[int],
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "LN",
        _class: str = "LN",
        **kwargs,
    ) -> None:
        super().__init__(
            output_dim,
            full_latent_dim,
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

        self.mu = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )
        self.log_var = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mu.save_origin_state()
        self.log_var.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss()
            + self.mu.deviation_loss()
            + self.log_var.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    @staticmethod
    def log_likelihood(
        x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        return -0.5 * (
            torch.square(x - mu) / torch.exp(log_var) + math.log(2 * math.pi) + log_var
        )

    def forward(
        self, full_x: typing.Tuple[torch.Tensor], feed_dict: typing.Mapping
    ) -> torch.Tensor:
        x = self.mlp(full_x)
        mu = torch.expm1(self.mu(x))
        log_var = self.log_var(x)
        return mu, log_var

    def loss(
        self,
        mu_var: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        mu, log_var = mu_var

        raw_loss = -self.log_likelihood(torch.log1p(y), mu, log_var).mean()
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] += (
            raw_loss.item() * mu.shape[0]
        )
        reg_loss = raw_loss
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] += (
            reg_loss.item() * mu.shape[0]
        )

        if self.fine_tune:
            return reg_loss + self.deviation_reg * self.deviation_loss()
        else:
            return reg_loss

    def init_loss_record(self, loss_record: typing.Mapping) -> None:
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] = 0
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] = 0


class ZILN(LN):
    r"""
    Build a Zero-Inflated Log Normal generative module.

    Parameters
    ----------
    output_dim
        Dimensionality of the output tensor.
    full_latent_dim
        Dimensionality of the latent variable and Numbers of batches.
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    lambda_reg
        NOT USED.
    fine_tune
        Whether the module is used in fine-tuning.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """

    def __init__(
        self,
        output_dim: int,
        full_latent_dim: typing.Tuple[int],
        h_dim: int = 128,
        depth: int = 1,
        dropout: float = 0.0,
        lambda_reg: float = 0.0,
        fine_tune: bool = False,
        deviation_reg: float = 0.0,
        name: str = "ZILN",
        _class: str = "ZILN",
        **kwargs,
    ) -> None:
        super().__init__(
            output_dim,
            full_latent_dim,
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

        self.pi = (
            Linear(h_dim, output_dim)
            if depth > 0
            else Linear(full_latent_dim, output_dim)
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.mlp.save_origin_state()
        self.mu.save_origin_state()
        self.log_var.save_origin_state()
        self.pi.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        return self.deviation_reg * (
            self.mlp.deviation_loss()
            + self.mu.deviation_loss()
            + self.log_var.deviation_loss()
            + self.pi.deviation_loss()
        )

    # fine_tune
    def check_fine_tune(self) -> None:
        if self.fine_tune:
            self.save_origin_state()

    @staticmethod
    def log_likelihood(
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        pi: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        case_zero = -F.softplus(-pi)
        case_non_zero = (
            -pi
            - F.softplus(-pi)
            - 0.5
            * (
                torch.square(x - mu) / torch.exp(log_var)
                + math.log(2 * math.pi)
                + log_var
            )
        )
        mask = (x < eps).float()
        res = mask * case_zero + (1 - mask) * case_non_zero
        return res

    def forward(
        self, full_x: typing.Tuple[torch.Tensor], feed_dict: typing.Mapping
    ) -> torch.Tensor:
        x = self.mlp(full_x)
        mu = torch.expm1(self.mu(x))
        log_var = self.log_var(x)
        pi = self.pi(x)
        return mu, log_var, pi

    def loss(
        self,
        mu_var_pi: typing.Tuple[torch.Tensor],
        feed_dict: typing.Mapping,
        loss_record: typing.Mapping,
    ) -> torch.Tensor:
        y = feed_dict["exprs"]
        mu, log_var, pi = mu_var_pi

        raw_loss = -self.log_likelihood(torch.log1p(y), mu, log_var, pi).mean()
        loss_record[self.record_prefix + "/" + self.name + "/raw_loss"] += (
            raw_loss.item() * mu.shape[0]
        )
        reg_loss = raw_loss
        loss_record[self.record_prefix + "/" + self.name + "/regularized_loss"] += (
            reg_loss.item() * mu.shape[0]
        )

        if self.fine_tune:
            return reg_loss + self.deviation_reg * self.deviation_loss()
        else:
            return reg_loss


class MSE(ProbModel):
    def __init__(self, *args, **kwargs):
        utils.logger.warning(
            "Prob module `MSE` is no longer supported, running as `ProbModel`"
        )
        super().__init__(*args, **kwargs)
