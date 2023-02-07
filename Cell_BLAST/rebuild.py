r"""
Rebuild basic NN components in PyTorch to follows TensorFlow behaviors
"""

import typing

import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    # allow multiple input tensors
    # add initialize standard deviation
    # add truncating option
    def __init__(
        self,
        in_features: typing.Union[typing.Tuple[int], int],
        out_features: int,
        bias: bool = True,
        init_std: float = 0.01,
        trunc: bool = True,
    ) -> "Linear":
        if not isinstance(in_features, list) and not isinstance(in_features, tuple):
            in_features = [in_features]

        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.ParameterList()
        for _in_features in in_features:
            self.weight.append(nn.Parameter(torch.Tensor(out_features, _in_features)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.init_std = init_std
        self.trunc = trunc
        self.reset_parameters()

    def reset_parameters(self):
        # change initializers
        if self.trunc:
            for _weight in self.weight:
                nn.init.trunc_normal_(
                    _weight,
                    std=self.init_std,
                    a=-2 * self.init_std,
                    b=2 * self.init_std,
                )
        else:
            for _weight in self.weight:
                nn.init.normal_(_weight, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, input: typing.Union[typing.Tuple[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if not isinstance(input, list) and not isinstance(input, tuple):
            input = [input]
        for i, (_input, _weight) in enumerate(zip(input, self.weight)):
            if i:
                result = result + F.linear(_input, _weight, None)
            else:
                result = F.linear(_input, _weight, self.bias)
        return result
        # return F.linear(torch.cat(input, dim=-1), torch.cat(list(self.weight), dim=-1), self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    # fine-tune
    def save_origin_state(self) -> None:
        self.weight_origin = [_weight.detach().clone() for _weight in self.weight]
        if self.bias is not None:
            self.bias_origin = self.bias.detach().clone()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        for i, (_weight, _weight_origin) in enumerate(
            zip(self.weight, self.weight_origin)
        ):
            if i:
                result = result + F.mse_loss(_weight, _weight_origin)
            else:
                result = F.mse_loss(_weight, _weight_origin)
        if self.bias is not None:
            result = result + F.mse_loss(self.bias, self.bias_origin)
        return result


class RMSprop(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.9,
        eps=1e-10,
        weight_decay=0,
        momentum=0.0,
        centered=False,
        decoupled_decay=False,
        lr_in_momentum=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            decoupled_decay=decoupled_decay,
            lr_in_momentum=lr_in_momentum,
        )

        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.ones_like(p)  # PyTorch inits to zero
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p)

                square_avg = state["square_avg"]
                one_minus_alpha = 1.0 - group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if group["decoupled_decay"]:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)  # PyTorch original

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1)
                        .add(group["eps"])
                        .sqrt_()
                    )  # eps in sqrt
                    # grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)  # PyTorch original
                else:
                    avg = square_avg.add(group["eps"]).sqrt_()  # eps moved in sqrt

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if group["lr_in_momentum"]:
                        buf.mul_(group["momentum"]).addcdiv_(
                            grad, avg, value=group["lr"]
                        )
                        p.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                        p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])

        return loss


class MLP(nn.Module):
    def __init__(
        self,
        i_dim: typing.Tuple[int],
        o_dim: typing.Tuple[int],
        dropout: typing.Tuple[float],
        bias: bool = True,
        batch_normalization: bool = False,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.dropout = dropout
        self.bias = bias
        self.batch_normalization = batch_normalization
        self.activation = activation

        self.hiddens = nn.ModuleList()
        module_seq = []

        for _i_dim, _o_dim, _dropout in zip(i_dim, o_dim, dropout):
            if _dropout > 0:
                module_seq.append(nn.Dropout(dropout))

            hidden = Linear(_i_dim, _o_dim, bias=bias)
            self.hiddens.append(hidden)
            module_seq.append(hidden)

            if batch_normalization:
                module_seq.append(nn.BatchNorm1d(_o_dim, eps=0.001, momentum=0.01))

            if activation:
                module_seq.append(nn.LeakyReLU(0.2))

        self.layer_seq = nn.Sequential(*module_seq)

        self.first_layer_trainable = True

    def forward(self, x: torch.Tensor):
        return self.layer_seq(x)

    # fine-tune
    def save_origin_state(self) -> None:
        for _hidden in self.hiddens:
            _hidden.save_origin_state()

    # fine-tune
    def deviation_loss(self) -> torch.Tensor:
        loss = torch.tensor(0)
        for _hidden in self.hiddens:
            loss = loss + _hidden.deviation_loss()
        return loss

    # fine-tune
    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if not self.first_layer_trainable:
            if len(self.hiddens) > 0:
                self.hiddens[0].requires_grad_(False)
        return self

    @property
    def first_layer_trainable(self):
        return self._first_layer_trainable

    @first_layer_trainable.setter
    def first_layer_trainable(self, flag: bool):
        self._first_layer_trainable = flag
        if len(self.hiddens) > 0:
            self.hiddens[0].requires_grad_(flag)
