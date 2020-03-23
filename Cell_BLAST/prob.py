r"""
Probabilistic / decoder modules for DIRECTi
"""

import typing
import abc

import numpy as np
import tensorflow as tf

from . import module, nn


class ProbModel(module.Module):
    r"""
    Abstract base class for generative model modules.
    """
    def __init__(
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name: str = "ProbModel"
    ) -> None:
        super(ProbModel, self).__init__(name=name)
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.fine_tune = fine_tune
        self.deviation_reg = deviation_reg
        self.deviation_regularizer = \
            (lambda x: self.deviation_reg * tf.reduce_mean(tf.square(x))) \
            if self.fine_tune and self.deviation_reg > 0 else None

    @staticmethod
    def _normalize(  # pylint: disable=unused-argument
            x: typing.Union[np.ndarray, tf.Tensor],
            library_size: typing.Union[np.ndarray, tf.Tensor]
    ) -> typing.Union[np.ndarray, tf.Tensor]:  # pragma: no cover
        return x

    @staticmethod
    def _add_noise(  # pylint: disable=unused-argument
            x: typing.Union[np.ndarray, tf.Tensor],
            random_state: typing.Optional[np.random.RandomState] = None
    ) -> typing.Union[np.ndarray, tf.Tensor]:  # pragma: no cover
        return x

    @staticmethod
    def _preprocess(x: tf.Tensor) -> tf.Tensor:
        return x

    def _loss(
            self, ref: tf.Tensor, latent: tf.Tensor, training_flag: tf.Tensor,
            tail_concat: typing.Optional[typing.List[tf.Tensor]] = None,
            scope: str = "decoder"
    ) -> tf.Tensor:
        with tf.variable_scope(f"{scope}/{self.scope_safe_name}"):
            dropout = np.zeros(self.depth)
            dropout[1:] = self.dropout  # No dropout for first layer
            mlp_kwargs = dict(
                dropout=dropout.tolist(), dense_kwargs=dict(
                    deviation_regularizer=self.deviation_regularizer
                ), training_flag=training_flag
            )
            ptr = nn.mlp(latent, [self.h_dim] * self.depth, **mlp_kwargs)
            ptr = (ptr if isinstance(ptr, list) else [ptr]) + (tail_concat or [])
            self.log_likelihood = self._log_likelihood(ref, ptr)
            self.mean_log_likelihood = tf.reduce_mean(self.log_likelihood, axis=1)  # feature size invariant
            raw_loss = tf.negative(tf.reduce_mean(self.mean_log_likelihood), name="raw_loss")
            regularized_loss = tf.add(
                raw_loss, self.lambda_reg * self._build_regularizer(),
                name="regularized_loss"
            )
        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, f"{scope}/{self.scope_safe_name}")
        tf.add_to_collection(tf.GraphKeys.LOSSES, raw_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, regularized_loss)
        return regularized_loss

    @abc.abstractmethod
    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _build_regularizer(self) -> tf.Tensor:
        return 0

    def _get_config(self) -> typing.Mapping:
        return {
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "fine_tune": self.fine_tune,
            "deviation_reg": self.deviation_reg,
            **super(ProbModel, self)._get_config()
        }

    def __bool__(self) -> bool:
        return True


class CountBased(ProbModel):  # pylint: disable=abstract-method

    @staticmethod
    def _normalize(
            x: typing.Union[np.ndarray, tf.Tensor],
            library_size: typing.Union[np.ndarray, tf.Tensor]
    ) -> typing.Union[tf.Tensor]:
        return x / (library_size / 10000)

    @staticmethod
    def _add_noise(
            x: typing.Union[np.ndarray, tf.Tensor],
            random_state: typing.Optional[np.random.RandomState] = None
    ) -> typing.Union[np.ndarray, tf.Tensor]:
        if random_state is None:
            return tf.squeeze(tf.random_poisson(x, [1]), axis=0)
        else:
            return random_state.poisson(x)

    @staticmethod
    def _preprocess(x: tf.Tensor) -> tf.Tensor:
        return tf.log1p(x)


class NB(CountBased):  # Negative binomial
    r"""
    Build a Negative Binomial generative module.

    Parameters
    ----------
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    fine_tune
        Whether the module is used in fine-tuning.
    lambda_reg
        Regularization strength for the generative model parameters.
        Here log-scale variance of the scale parameter
        is regularized to improve numerical stability.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """
    def __init__(
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name: str = "NB"
    ) -> None:
        super(NB, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:
        recon_dim = ref.get_shape().as_list()[1]
        self.softmax_mu = tf.nn.softmax(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="softmax_mu_dense"
        ), name="softmax_mu")
        self.log_theta = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="log_theta_dense"
        ), name="log_theta")
        self.recon = mu = \
            self.softmax_mu * tf.reduce_sum(ref, axis=1, keepdims=True)
        return self._log_nb_positive(ref, mu, self.log_theta)

    def _build_regularizer(self) -> tf.Tensor:
        with tf.name_scope("regularization"):
            return tf.nn.moments(self.log_theta, axes=[0, 1])[1]

    @staticmethod
    def _log_nb_positive(
            x: tf.Tensor, mu: tf.Tensor, log_theta: tf.Tensor,
            eps: float = 1e-8
    ) -> tf.Tensor:
        with tf.name_scope("log_nb_positive"):
            theta = tf.exp(log_theta)
            return theta * log_theta \
                - theta * tf.log(theta + mu + eps) \
                + x * tf.log(mu + eps) - x * tf.log(theta + mu + eps) \
                + tf.lgamma(x + theta) - tf.lgamma(theta) \
                - tf.lgamma(x + 1)


class ZINB(NB):  # Zero-inflated negative binomial
    r"""
    Build a Zero-Inflated Negative Binomial generative module.

    Parameters
    ----------
    h_dim
        Dimensionality of the hidden layers in the decoder MLP.
    depth
        Number of hidden layers in the decoder MLP.
    dropout
        Dropout rate.
    fine_tune
        Whether the module is used in fine-tuning.
    lambda_reg
        Regularization strength for the generative model parameters.
        Here log-scale variance of the scale parameter
        is regularized to improve numerical stability.
    deviation_reg
        Regularization strength for the deviation from original model weights.
    name
        Name of the module.
    """
    def __init__(
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name: str = "ZINB"
    ) -> None:
        super(ZINB, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:
        recon_dim = ref.get_shape().as_list()[1]
        self.softmax_mu = tf.nn.softmax(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="softmax_mu_dense"
        ), name="softmax_mu")
        self.log_theta = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="log_theta_dense"
        ), name="log_theta")
        self.pi = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="dropout_logit_dense"
        ), name="dropout_logit")
        self.recon = mu = \
            self.softmax_mu * tf.reduce_sum(ref, axis=1, keepdims=True)
        self.dropout_rate = tf.sigmoid(self.pi)
        return self._log_zinb_positive(ref, mu, self.log_theta, self.pi)

    @staticmethod
    def _log_zinb_positive(
            x: tf.Tensor, mu: tf.Tensor, log_theta: tf.Tensor,
            pi: tf.Tensor, eps: float = 1e-8
    ) -> tf.Tensor:
        r"""
        From scVI
        """
        with tf.name_scope("log_zinb_positive"):
            theta = tf.exp(log_theta)
            with tf.name_scope("case_zero"):
                case_zero = tf.nn.softplus(
                    - pi + theta * log_theta -
                    theta * tf.log(theta + mu + eps)
                ) - tf.nn.softplus(- pi)
            with tf.name_scope("case_non_zero"):
                case_non_zero = - pi - tf.nn.softplus(- pi) \
                    + theta * log_theta \
                    - theta * tf.log(theta + mu + eps) \
                    + x * tf.log(mu + eps) - x * tf.log(theta + mu + eps) \
                    + tf.lgamma(x + theta) - tf.lgamma(theta) \
                    - tf.lgamma(x + 1)
            with tf.name_scope("mixture"):
                mask = tf.cast(tf.less(x, eps), tf.float32)
                res = tf.identity(
                    tf.multiply(mask, case_zero) +
                    tf.multiply(1 - mask, case_non_zero),
                    name="likelihood")
            return res


class LN(CountBased):
    r"""
    Build a Log Normal generative module.

    Parameters
    ----------
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
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name: str = "LN"
    ) -> None:
        super(LN, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:
        recon_dim = ref.get_shape().as_list()[1]
        self.mu = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="mu_dense"
        ), name="mu")
        self.log_var = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="log_var_dense"
        ), name="log_var")
        self.recon = tf.expm1(self.mu)
        return self._log_ln_positive(
            tf.log1p(ref), self.mu, self.log_var)

    @staticmethod
    def _log_ln_positive(
            x: tf.Tensor, mu: tf.Tensor, log_var: tf.Tensor
    ) -> tf.Tensor:
        with tf.name_scope("log_ln"):
            return - 0.5 * (
                tf.square(x - mu) / tf.exp(log_var)
                + tf.log(2 * np.pi) + log_var
            )


class ZILN(LN):
    r"""
    Build a Zero-Inflated Log Normal generative module.

    Parameters
    ----------
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
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name: str = "ZILN"
    ) -> None:
        super(ZILN, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:
        recon_dim = ref.get_shape().as_list()[1]
        self.mu = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="mu_dense"
        ), name="mu")
        self.log_var = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="log_var_dense"
        ), name="log_var")
        self.pi = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="dropout_logit_dense"
        ), name="dropout_logit")
        self.recon = tf.expm1(self.mu)
        self.dropout_rate = tf.sigmoid(self.pi)
        return self._log_ziln_positive(
            tf.log1p(ref), self.mu, self.log_var, self.pi)

    @staticmethod
    def _log_ziln_positive(
            x: tf.Tensor, mu: tf.Tensor, log_var: tf.Tensor,
            pi: tf.Tensor, eps: float = 1e-8
    ) -> tf.Tensor:
        with tf.name_scope("log_ziln"):
            with tf.name_scope("case_zero"):
                case_zero = - tf.nn.softplus(- pi)
            with tf.name_scope("case_non_zero"):
                case_non_zero = - pi - tf.nn.softplus(- pi) - 0.5 * (
                    tf.square(x - mu) / tf.exp(log_var)
                    + tf.log(2 * np.pi) + log_var
                )
            with tf.name_scope("mixture"):
                mask = tf.cast(tf.less(x, eps), tf.float32)
                res = tf.identity(
                    tf.multiply(mask, case_zero) +
                    tf.multiply(1 - mask, case_non_zero),
                    name="likelihood"
                )
            return res


class MSE(ProbModel):

    def __init__(
            self, h_dim: int = 128, depth: int = 1, dropout: float = 0.0,
            lambda_reg: float = 0.0, fine_tune: bool = False,
            deviation_reg: float = 0.0, name="MSE"
    ) -> None:
        super(MSE, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(
            self, ref: tf.Tensor, pre_recon: typing.List[tf.Tensor]
    ) -> tf.Tensor:
        recon_dim = ref.get_shape().as_list()[1]
        self.mu = tf.identity(nn.dense(
            pre_recon, recon_dim,
            deviation_regularizer=self.deviation_regularizer,
            scope="mu_dense"
        ), name="mu")
        return tf.negative(tf.square(ref - self.mu))
