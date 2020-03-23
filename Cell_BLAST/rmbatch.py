r"""
Batch effect removing modules for DIRECTi
"""

import typing

import numpy as np
import tensorflow as tf

from . import module, nn, utils


class RMBatch(module.Module):
    r"""
    Parent class for systematical bias / batch effect removal modules.
    """
    def __init__(
            self, batch_dim: int, delay: int = 20, name: str = "RMBatch"
    ) -> None:
        super(RMBatch, self).__init__(name=name)
        self.batch_dim = batch_dim
        self.delay = delay
        if self._delay_guard not in self.on_epoch_end:
            self.on_epoch_end.append(self._delay_guard)

    def _build_regularizer(  # pylint: disable=unused-argument
            self, input_tensor: tf.Tensor, training_flag: tf.Tensor,
            epoch: tf.Tensor, scope: str = ""
    ) -> tf.Tensor:
        with tf.name_scope("placeholder/"):
            self.batch = tf.placeholder(
                dtype=tf.float32, shape=(None, self.batch_dim),
                name=self.scope_safe_name
            )
        return 0.0

    def _build_feed_dict(self, data_dict: utils.DataDict) -> typing.Mapping:
        return {
            self.batch: utils.densify(data_dict[self.name])
        } if self.name in data_dict else {}

    def __bool__(self) -> bool:
        return True

    def _get_config(self) -> typing.Mapping:
        return {
            "batch_dim": self.batch_dim,
            "delay": self.delay,
            **super(RMBatch, self)._get_config()
        }

    def _delay_guard(  # pylint: disable=unused-argument
            self, model: "directi.DIRECTi",
            train_data_dict: utils.DataDict,
            val_data_dict: utils.DataDict,
            loss: tf.Tensor
    ) -> bool:
        _epoch = model.sess.run(model.epoch)
        return _epoch >= self.delay


class Adversarial(RMBatch):
    r"""
    Build a batch effect correction module that uses adversarial batch alignment.

    Parameters
    ----------
    batch_dim
        Number of batches.
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
    name
        Name of the module.
    """
    def __init__(
            self, batch_dim: int, h_dim: int = 128, depth: int = 1,
            dropout: float = 0.0, lambda_reg: float = 0.01,
            n_steps: int = 1, delay: int = 20, name: str = "AdvBatch"
    ) -> None:
        super(Adversarial, self).__init__(batch_dim, delay=delay, name=name)
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.n_steps = n_steps

    def _build_regularizer(
            self, input_tensor: tf.Tensor, training_flag: tf.Tensor,
            epoch: tf.Tensor, scope: str = "discriminator"
    ) -> tf.Tensor:
        with tf.name_scope("placeholder/"):
            self.batch = tf.placeholder(
                dtype=tf.float32, shape=(None, self.batch_dim),
                name=self.scope_safe_name
            )
        self.build_regularizer_scope = f"{scope}/{self.scope_safe_name}"
        with tf.variable_scope(self.build_regularizer_scope):
            mask = tf.cast(tf.reduce_sum(self.batch, axis=1) > 0, tf.int32)
            batch = tf.dynamic_partition(self.batch, mask, 2)[1]
            input_tensor = tf.dynamic_partition(input_tensor, mask, 2)[1]
            dropout = np.zeros(self.depth)
            dropout[1:] = self.dropout  # No dropout for first layer
            batch_pred = tf.identity(nn.dense(nn.mlp(
                input_tensor, [self.h_dim] * self.depth,
                dropout=dropout.tolist(), training_flag=training_flag
            ), self.batch_dim), "batch_logit")
            self.batch_d_loss = tf.cast(
                epoch >= self.delay, tf.float32
            ) * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=batch, logits=batch_pred
                ), name="d_loss"
            )
            self.batch_g_loss = tf.negative(self.batch_d_loss, name="g_loss")

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.batch_d_loss)
        return self.lambda_reg * self.batch_g_loss

    def _compile(self, optimizer: str, lr: float) -> None:
        with tf.variable_scope(f"optimize/{self.scope_safe_name}"):
            optimizer = getattr(tf.train, optimizer)(lr)
            control_dependencies = []
            for _ in range(self.n_steps):
                with tf.control_dependencies(control_dependencies):
                    self.step = optimizer.minimize(
                        self.lambda_reg * self.batch_d_loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            self.build_regularizer_scope
                        )
                    )
                    control_dependencies = [self.step]
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.step)

    def _get_config(self) -> typing.Mapping:
        return {
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "n_steps": self.n_steps,
            **super(Adversarial, self)._get_config()
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
            self, batch_dim: int, n_neighbors: int = 5,
            lambda_reg: float = 1.0, delay: int = 20, name: str = "MNN"
    ) -> None:
        super(MNN, self).__init__(batch_dim, delay=delay, name=name)
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg

    def _build_regularizer(
            self, input_tensor: tf.Tensor, training_flag: tf.Tensor,
            epoch: tf.Tensor, scope: str = "MNN"
    ) -> tf.Tensor:
        with tf.name_scope("placeholder/"):
            self.batch = tf.placeholder(dtype=tf.float32, shape=(
                None, self.batch_dim
            ), name=self.scope_safe_name)
        with tf.name_scope(f"{scope}/{self.scope_safe_name}"):
            batches = tf.dynamic_partition(
                input_tensor,
                partitions=tf.argmax(self.batch, axis=1, output_type=tf.int32),
                num_partitions=self.batch_dim
            )
            use_flags = [tf.shape(batch)[0] > 0 for batch in batches]
            penalties = []
            for i in range(len(batches)):
                for j in range(i + 1, len(batches)):
                    penalties.append(tf.cond(
                        tf.logical_and(use_flags[i], use_flags[j]),
                        lambda i=i, j=j: self._cross_batch_penalty(batches[i], batches[j]),
                        lambda: tf.zeros((0,))
                    ))
            penalties = tf.concat(penalties, axis=0)
            return tf.cast(
                epoch > self.delay, tf.float32
            ) * self.lambda_reg * tf.reduce_mean(penalties, name="MNN_loss")

    def _cross_batch_penalty(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:  # MNN
        x1, y0 = tf.expand_dims(x, axis=1), tf.expand_dims(y, axis=0)
        xy_dist = tf.reduce_sum(tf.square(x1 - y0), axis=2)
        xy_mask = tf.cast(self._mnn_mask(xy_dist, self.n_neighbors), tf.float32)
        return tf.reshape(xy_dist * xy_mask, [-1])

    @staticmethod
    def _neighbor_mask(d: tf.Tensor, k: int) -> tf.Tensor:
        n = tf.shape(d)[1]
        _, idx = tf.nn.top_k(tf.negative(d), k=tf.minimum(k, n))
        return tf.cast(tf.reduce_sum(tf.one_hot(idx, depth=n), axis=1), tf.bool)

    @staticmethod
    def _mnn_mask(d: tf.Tensor, k: int) -> tf.Tensor:
        return tf.logical_and(
            MNN._neighbor_mask(d, k),
            tf.transpose(MNN._neighbor_mask(tf.transpose(d), k))
        )

    def _get_config(self) -> typing.Mapping:
        return {
            "n_neighbors": self.n_neighbors,
            "lambda_reg": self.lambda_reg,
            **super(MNN, self)._get_config()
        }


class MNNAdversarial(Adversarial, MNN):
    r"""
    Build a batch effect correction module that uses adversarial batch alignment
    among cells with mutual nearest neighbors.

    Parameters
    ----------
    batch_dim
        Number of batches.
    h_dim
        Dimensionality of the hidden layers in the discriminator MLP.
    depth
        Number of hidden layers in the discriminator MLP.
    dropout
        Dropout rate.
    lambda_reg
        Strength of batch effect correction.
    n_steps
        How many discriminator steps to run for each encoder step.
    n_neighbors
        Number of nearest neighbors to use when selecting mutual nearest
        neighbors.
    delay
        How many epoches to delay before using MNN batch correction.
    name
        Name of the module.
    """

    def __init__(
            self, batch_dim: int, h_dim: int = 128, depth: int = 1,
            dropout: float = 0.0, lambda_reg: float = 0.01, n_steps: int = 1,
            n_neighbors: int = 5, delay: int = 20, name="MNNAdvBatch"
    ) -> None:
        super(MNNAdversarial, self).__init__(
            batch_dim, h_dim, depth, dropout, lambda_reg, n_steps,
            delay=delay, name=name
        )  # Calls Adversarial.__init__
        self.n_neighbors = n_neighbors

    def _build_regularizer(
            self, input_tensor: tf.Tensor, training_flag: tf.Tensor,
            epoch: tf.Tensor, scope: str = "discriminator"
    ) -> tf.Tensor:
        with tf.name_scope("placeholder/"):
            self.batch = tf.placeholder(
                dtype=tf.float32, shape=(None, self.batch_dim),
                name=self.scope_safe_name
            )
        self.build_regularizer_scope = f"{scope}/{self.scope_safe_name}"
        with tf.variable_scope(self.build_regularizer_scope):
            mask = tf.cast(tf.reduce_sum(self.batch, axis=1) > 0, tf.int32)
            batch = tf.dynamic_partition(self.batch, mask, 2)[1]
            input_tensor = tf.dynamic_partition(input_tensor, mask, 2)[1]
            input_idx = tf.expand_dims(tf.cast(
                tf.range(tf.shape(input_tensor)[0]), tf.float32
            ), axis=1)
            _input_tensor = tf.concat([input_idx, input_tensor], axis=1)
            batches = tf.dynamic_partition(
                _input_tensor,
                partitions=tf.argmax(batch, axis=1, output_type=tf.int32),
                num_partitions=self.batch_dim
            )
            use_flags = [tf.shape(item)[0] > 0 for item in batches]
            batches = [(item[:, 0], item[:, 1:]) for item in batches]
            include_idx = []
            for i in range(len(batches)):
                for j in range(i + 1, len(batches)):
                    include_idx.append(tf.cond(
                        tf.logical_and(use_flags[i], use_flags[j]),
                        lambda i=i, j=j: self._mnn_idx(batches[i], batches[j], self.n_neighbors),
                        lambda: (tf.zeros((0,)), tf.zeros((0,)))
                    ))
            include_idx = [j for i in include_idx for j in i]  # flatten
            self.include_idx = tf.unique(tf.cast(
                tf.concat(include_idx, axis=0), tf.int32))[0]
            input_tensor = tf.gather(input_tensor, self.include_idx)
            batch = tf.gather(batch, self.include_idx)
            dropout = np.zeros(self.depth)
            dropout[1:] = self.dropout  # No dropout for first layer
            batch_pred = tf.identity(nn.dense(nn.mlp(
                input_tensor, [self.h_dim] * self.depth,
                dropout=dropout.tolist(), training_flag=training_flag
            ), self.batch_dim), "batch_logit")
            self.batch_d_loss = tf.multiply(tf.cast(
                epoch >= self.delay, tf.float32
            ), tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=batch, logits=batch_pred
                ), name="raw_d_loss"
            ), name="d_loss")
            self.batch_g_loss = tf.negative(self.batch_d_loss, name="g_loss")

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.batch_d_loss)
        return self.lambda_reg * self.batch_g_loss

    @staticmethod
    def _mnn_idx(
            batch1: typing.Tuple[tf.Tensor, tf.Tensor],
            batch2: typing.Tuple[tf.Tensor, tf.Tensor],
            k: int
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        (xi, x), (yi, y) = batch1, batch2
        x1, y0 = tf.expand_dims(x, axis=1), tf.expand_dims(y, axis=0)
        xy_dist = tf.reduce_sum(tf.square(x1 - y0), axis=2)
        xy_mask = tf.cast(MNNAdversarial._mnn_mask(xy_dist, k), tf.int32)
        return (
            tf.dynamic_partition(xi, tf.cast(
                tf.reduce_sum(xy_mask, axis=1) > 0, tf.int32
            ), 2)[1],
            tf.dynamic_partition(yi, tf.cast(
                tf.reduce_sum(xy_mask, axis=0) > 0, tf.int32
            ), 2)[1]
        )


# EXPERIMENTAL
class AdaptiveMNNAdversarial(MNNAdversarial):

    def __init__(
            self, batch_dim: int, h_dim: int = 128, depth: int = 1,
            dropout: float = 0.0, lambda_reg: float = 0.01, n_steps: int = 1,
            n_neighbors: int = 5, delay: int = 20, name: str = "AdptMNNAdvBatch"
    ) -> None:
        super(AdaptiveMNNAdversarial, self).__init__(
            batch_dim, h_dim, depth, dropout, lambda_reg, n_steps, n_neighbors,
            delay=delay, name=name
        )

    def _build_regularizer(
            self, input_tensor: tf.Tensor, training_flag: tf.Tensor,
            epoch: tf.Tensor, scope: str = "discriminator"
    ) -> tf.Tensor:
        with tf.name_scope("placeholder/"):
            self.batch = tf.placeholder(
                dtype=tf.float32, shape=(None, self.batch_dim),
                name=self.scope_safe_name
            )
        self.build_regularizer_scope = f"{scope}/{self.scope_safe_name}"
        with tf.variable_scope(self.build_regularizer_scope):
            # Select cells with batch identity
            mask = tf.cast(tf.reduce_sum(self.batch, axis=1) > 0, tf.int32)
            batch = tf.dynamic_partition(self.batch, mask, 2)[1]
            input_tensor = tf.dynamic_partition(input_tensor, mask, 2)[1]
            # Build MNN mask
            n = tf.shape(batch)[0]
            input_idx = tf.expand_dims(tf.cast(tf.range(n), tf.float32), axis=1)
            _input_tensor = tf.concat([input_idx, input_tensor], axis=1)
            batches = tf.dynamic_partition(
                _input_tensor,
                partitions=tf.argmax(batch, axis=1, output_type=tf.int32),
                num_partitions=self.batch_dim
            )
            use_flags = [tf.shape(item)[0] > 0 for item in batches]
            batches = [(item[:, 0], item[:, 1:]) for item in batches]
            self.mask_mat = []
            for i in range(len(batches)):
                for j in range(i + 1, len(batches)):
                    idx_mask = tf.cond(
                        tf.logical_and(use_flags[i], use_flags[j]),
                        lambda i=i, j=j: self._mnn_idx_mask(
                            batches[i], batches[j], self.n_neighbors, n),
                        lambda: tf.zeros((n,))
                    )
                    idx_mask = tf.expand_dims(idx_mask, axis=1)
                    self.mask_mat.append(tf.concat([
                        tf.zeros((n, i)), idx_mask,
                        tf.zeros((n, j - i - 1)), idx_mask,
                        tf.zeros((n, self.batch_dim - j - 1))
                    ], axis=1))
            self.mask_mat = tf.cast(tf.add_n(self.mask_mat) > 0, tf.int32)
            include_mask = tf.cast(tf.reduce_sum(
                self.mask_mat, axis=1
            ) > 0, tf.int32)
            self.mask_mat = tf.dynamic_partition(self.mask_mat, include_mask, 2)[1]
            batch = tf.dynamic_partition(batch, include_mask, 2)[1]
            input_tensor = tf.dynamic_partition(input_tensor, include_mask, 2)[1]
            # Distriminator loss
            dropout = np.zeros(self.depth)
            dropout[1:] = self.dropout  # No dropout for first layer
            batch_pred = tf.identity(nn.dense(nn.mlp(
                input_tensor, [self.h_dim] * self.depth,
                dropout=dropout.tolist(), training_flag=training_flag
            ), self.batch_dim), "batch_logit")
            self.batch_d_loss = tf.cast(
                epoch >= self.delay, tf.float32
            ) * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=batch, logits=batch_pred
                ), name="d_loss"
            )
            # Generator loss
            self.batch_g_loss = tf.cast(
                epoch >= self.delay, tf.float32
            ) * tf.negative(tf.reduce_mean(tf.scan(
                self._masked_softmax_cross_entropy_with_logits,
                (batch, batch_pred, self.mask_mat),
                tf.zeros(()), parallel_iterations=128
            )), name="g_loss")

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.batch_d_loss)
        return self.lambda_reg * self.batch_g_loss

    @staticmethod
    def _mnn_idx_mask(
            batch1: typing.Tuple[tf.Tensor, tf.Tensor],
            batch2: typing.Tuple[tf.Tensor, tf.Tensor],
            k: int, n: int
    ) -> tf.Tensor:
        idx1, idx2 = AdaptiveMNNAdversarial._mnn_idx(batch1, batch2, k)
        idx = tf.cast(tf.concat([idx1, idx2], axis=0), tf.int32)
        return tf.reduce_sum(tf.one_hot(idx, depth=n), axis=0)

    @staticmethod
    def _masked_softmax_cross_entropy_with_logits(
            cum: tf.Tensor,  # pylint: disable=unused-argument
            tensors: typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        labels, logits, mask = tensors
        labels = tf.dynamic_partition(labels, mask, 2)[1]
        logits = tf.dynamic_partition(logits, mask, 2)[1]
        return tf.reduce_sum(labels * (tf.reduce_logsumexp(logits) - logits))
