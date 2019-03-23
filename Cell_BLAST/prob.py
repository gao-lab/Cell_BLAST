"""
Probabilistic / decoder modules for DIRECTi
"""


import numpy as np
import tensorflow as tf
from . import nn
from . import module


class ProbModel(module.Module):
    """
    Parent class for probabilistic model modules.
    """
    def __init__(self, h_dim=128, depth=1, dropout=0.0, lambda_reg=0.0,
                 fine_tune=False, deviation_reg=0.0, name="ProbModel"):
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
    def _add_noise(x):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `add_noise` from `ProbModel`!")

    @staticmethod
    def _preprocess(x):
        return x

    def _loss(self, ref, latent, training_flag,
              tail_concat=None, scope="decoder"):
        with tf.variable_scope("%s/%s" % (scope, self.scope_safe_name)):
            # if self.fine_tune:
            #     assert isinstance(latent, (list, tuple)) and len(latent) > 1
            #     dense_kwargs = [dict(
            #         weights_regularizer=[None] * (len(latent) - 1) + [
            #             self.deviation_regularizer]
            #     )] + [None] * (self.depth - 1)
            # else:
            #     dense_kwargs = None
            mlp_kwargs = dict(
                dropout=self.dropout, dense_kwargs=dict(
                    deviation_regularizer=self.deviation_regularizer
                ), training_flag=training_flag
            )
            ptr = nn.mlp(latent, [self.h_dim] * self.depth, **mlp_kwargs)
            if not (tail_concat is None or isinstance(tail_concat, (list, tuple))):
                tail_concat = [tail_concat]
            ptr = [ptr] + tail_concat if tail_concat is not None else [ptr]
            raw_loss = tf.negative(tf.reduce_mean(
                self._log_likelihood(ref, ptr)
            ), name="raw_loss")
            regularized_loss = tf.add(
                raw_loss, self.lambda_reg * self._build_regularizer(),
                name="regularized_loss"
            )
        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            "%s/%s" % (scope, self.scope_safe_name)
        )
        tf.add_to_collection(tf.GraphKeys.LOSSES, raw_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, regularized_loss)
        return regularized_loss

    def _log_likelihood(self, ref, pre_recon):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `likelihood` from `ProbModel`!")

    def _build_regularizer(self):
        return 0

    def _get_config(self):
        return {
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "fine_tune": self.fine_tune,
            "deviation_reg": self.deviation_reg,
            **super(ProbModel, self)._get_config()
        }

    def __bool__(self):
        return True


class NB(ProbModel):  # Negative binomial

    def __init__(self, h_dim=128, depth=1, dropout=0.0, lambda_reg=0.0,
                 fine_tune=False, deviation_reg=0.0, name="NB"):
        super(NB, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    @staticmethod
    def _add_noise(x):
        with tf.variable_scope("noise"):
            return tf.contrib.distributions.Poisson(rate=x).sample()

    @staticmethod
    def _preprocess(x):
        with tf.name_scope("preprocess"):
            return tf.log1p(x)

    def _log_likelihood(self, ref, pre_recon):
        recon_dim = ref.get_shape().as_list()[1]
        self.softmax_mu = tf.nn.softmax(nn.dense(
            pre_recon, recon_dim,
            # weights_trainable=not self.fine_tune,
            deviation_regularizer=self.deviation_regularizer,
            scope="softmax_mu_dense"
        ), name="softmax_mu")
        self.log_theta = tf.identity(nn.dense(
            pre_recon, recon_dim,
            # weights_trainable=not self.fine_tune,
            deviation_regularizer=self.deviation_regularizer,
            scope="log_theta_dense"
        ), name="log_theta")
        self.recon = mu = \
            self.softmax_mu * tf.reduce_sum(ref, axis=1, keepdims=True)
        return self._log_nb_positive(ref, mu, self.log_theta)

    def _build_regularizer(self):
        with tf.name_scope("regularization"):
            return tf.nn.moments(self.log_theta, axes=[0, 1])[1]

    @staticmethod
    def _log_nb_positive(x, mu, log_theta, eps=1e-8):
        with tf.name_scope("log_nb_positive"):
            theta = tf.exp(log_theta)
            return theta * log_theta \
                - theta * tf.log(theta + mu + eps) \
                + x * tf.log(mu + eps) - x * tf.log(theta + mu + eps) \
                + tf.lgamma(x + theta) - tf.lgamma(theta) \
                - tf.lgamma(x + 1)


class ZINB(NB):  # Zero-inflated negative binomial
    """
    Build a Zero-Inflated Negative Binomial module.

    Parameters
    ----------
    h_dim : int
        Dimensionality of hidden layers in MLP, by default 128.
    depth : int
        Number of hidden layers in MLP, by default 1.
    dropout : float
        Dropout rate, by default 0.0.
    lambda_reg : float
        Regularization strength for probabilistic model parameters,
        by default 0.0. Here log scale variance of the scale parameter
        is used as regularization to improve numerical stability.
    name : str
        Name of the module, by default "ZINB".
    """
    def __init__(self, h_dim=128, depth=1, dropout=0.0, lambda_reg=0.0,
                 fine_tune=False, deviation_reg=0.0, name="ZINB"):
        super(ZINB, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    def _log_likelihood(self, ref, pre_recon):
        recon_dim = ref.get_shape().as_list()[1]
        self.softmax_mu = tf.nn.softmax(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="softmax_mu_dense"
        ), name="softmax_mu")
        self.log_theta = tf.identity(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="log_theta_dense"
        ), name="log_theta")
        self.pi = tf.identity(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="dropout_logit_dense"
        ), name="dropout_logit")
        self.recon = mu = \
            self.softmax_mu * tf.reduce_sum(ref, axis=1, keepdims=True)
        self.dropout_rate = tf.sigmoid(self.pi)
        return self._log_zinb_positive(ref, mu, self.log_theta, self.pi)

    @staticmethod
    def _log_zinb_positive(x, mu, log_theta, pi, eps=1e-8):
        """
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


class ZIG(ProbModel):

    def __init__(self, h_dim=128, depth=1, dropout=0.0, lambda_reg=0.0,
                 fine_tune=False, deviation_reg=0.0, name="ZIG"):
        super(ZIG, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    @staticmethod
    def _add_noise(x):
        with tf.variable_scope("noise"):
            return tf.contrib.distributions.Poisson(rate=x).sample()

    @staticmethod
    def _preprocess(x):
        with tf.name_scope("preprocess"):
            return tf.log1p(x)

    def _log_likelihood(self, ref, pre_recon):
        recon_dim = ref.get_shape().as_list()[1]
        self.mu = tf.identity(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="mu_dense"
        ), name="mu")
        self.log_var = tf.identity(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="log_var_dense"
        ), name="log_var")
        self.pi = tf.identity(nn.dense(
            pre_recon, recon_dim,
            weights_trainable=not self.fine_tune,
            scope="dropout_logit_dense"
        ), name="dropout_logit")
        self.recon = tf.expm1(self.mu)
        self.dropout_rate = tf.sigmoid(self.pi)
        return self._log_zig_positive(
            tf.log1p(ref), self.mu, self.log_var, self.pi)

    @staticmethod
    def _log_zig_positive(x, mu, log_var, pi, eps=1e-8):
        with tf.name_scope("log_zig"):
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

    def __init__(self, h_dim=128, depth=1, dropout=0.0, lambda_reg=0.0,
                 fine_tune=False, deviation_reg=0.0, name="MSE"):
        super(MSE, self).__init__(
            h_dim, depth, dropout, lambda_reg,
            fine_tune, deviation_reg, name=name
        )

    @staticmethod
    def _add_noise(x):
        return x

    def _log_likelihood(self, ref, pre_recon):
        recon_dim = ref.get_shape().as_list()[1]
        self.mu = tf.identity(nn.dense(
            pre_recon, recon_dim, scope="mu_dense"
        ), name="mu")
        return tf.negative(tf.square(ref - self.mu))
