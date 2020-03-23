r"""
Neural network operations
"""

import typing

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer  # pylint: disable=no-name-in-module


def mlp(
        input_tensor: typing.Union[tf.Tensor, typing.List[tf.Tensor]],
        hidden_dim: int, activation: typing.Union[
            typing.Callable[[tf.Tensor], tf.Tensor],
            typing.List[typing.Callable[[tf.Tensor], tf.Tensor]]
        ] = tf.nn.leaky_relu,
        dropout: typing.Union[float, typing.List[float]] = 0.0,
        batch_normalization: typing.Union[bool, typing.List[bool]] = False,
        dense_kwargs: typing.Union[
            typing.Mapping, typing.List[typing.Mapping]
        ] = None,
        training_flag: typing.Optional[tf.Tensor] = None, scope: str = "mlp"
) -> tf.Tensor:
    if not isinstance(hidden_dim, list):
        hidden_dim = [hidden_dim]
    if isinstance(activation, list):
        assert len(activation) == len(hidden_dim)
    else:
        activation = [activation] * len(hidden_dim)
    if isinstance(dropout, list):
        assert len(dropout) == len(hidden_dim)
    else:
        dropout = [dropout] * len(hidden_dim)
    if isinstance(batch_normalization, list):
        assert len(batch_normalization) == len(hidden_dim)
    else:
        batch_normalization = [batch_normalization] * len(hidden_dim)
    if isinstance(dense_kwargs, list):
        assert len(dense_kwargs) == len(hidden_dim)
    else:
        dense_kwargs = [dense_kwargs] * len(hidden_dim)

    with tf.variable_scope(scope):
        ptr = input_tensor
        for l, (_hidden_dim, _activation, _dropout, _batch_normalization, _dense_kwargs) in enumerate(zip(
                hidden_dim, activation, dropout, batch_normalization, dense_kwargs
        )):
            assert not (_dropout or _batch_normalization) or training_flag is not None
            with tf.variable_scope(f"layer_{l}"):
                ptr = tf.layers.dropout(
                    ptr, rate=_dropout, training=training_flag
                ) if _dropout else ptr
                this_dense_kwargs = dict(
                    output_dim=_hidden_dim,
                    use_bias=not _batch_normalization
                )
                if _dense_kwargs is not None:
                    this_dense_kwargs.update(_dense_kwargs)
                ptr = dense(ptr, **this_dense_kwargs)
                ptr = tf.layers.batch_normalization(
                    ptr, center=True, scale=True, training=training_flag
                ) if _batch_normalization else ptr
                ptr = _activation(ptr) if _activation is not None else ptr
    return ptr


def dense(
        input_tensor: typing.Union[tf.Tensor, typing.List[tf.Tensor]],
        output_dim: int, use_bias: bool = True,
        weights_initializer: Initializer = tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer: Initializer = tf.zeros_initializer(),
        weights_trainable: bool = True, bias_trainable: bool = True,
        weights_regularizer: typing.Optional[
            typing.Callable[[tf.Tensor], tf.Tensor]
        ] = None,
        bias_regularizer: typing.Optional[
            typing.Callable[[tf.Tensor], tf.Tensor]
        ] = None,
        deviation_regularizer: typing.Optional[
            typing.Callable[[tf.Tensor], tf.Tensor]
        ] = None, scope: str = "dense"
) -> tf.Tensor:
    # `input_tensor` can accept a list of tensors instead of concatenating
    # them before passing to `dense`, which enables reusing part of the weight
    # matrix later.
    # Similarly, `weights_initializer`, `weights_trainable` and
    # `weights_regularizer` can also be a list, each element corresponding to
    # weights of an input tensor.
    with tf.variable_scope(scope):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if isinstance(weights_initializer, list):
            assert len(weights_initializer) == len(input_tensor)
        else:
            weights_initializer = [weights_initializer] * len(input_tensor)
        if isinstance(weights_trainable, list):
            assert len(weights_trainable) == len(input_tensor)
        else:
            weights_trainable = [weights_trainable] * len(input_tensor)
        if isinstance(weights_regularizer, list):
            assert len(weights_regularizer) == len(input_tensor)
        else:
            weights_regularizer = [weights_regularizer] * len(input_tensor)
        ptr = []
        for i, (
                _input_tensor, _weights_initializer,
                _weights_trainable, _weights_regularizer
        ) in enumerate(zip(
            input_tensor, weights_initializer,
            weights_trainable, weights_regularizer
        )):
            input_dim = _input_tensor.get_shape().as_list()[1]
            name = f"weights_{i}" if i > 0 else "weights"
            weights = tf.get_variable(
                name, shape=(input_dim, output_dim), dtype=tf.float32,
                initializer=_weights_initializer if deviation_regularizer is None
                else tf.zeros_initializer(),
                trainable=_weights_trainable
            )
            if _weights_regularizer is not None:
                reg = _weights_regularizer(weights)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            if deviation_regularizer is not None:
                name += "_ori"
                weights_ori = tf.get_variable(
                    name, shape=(input_dim, output_dim),
                    dtype=tf.float32, trainable=False
                )
                assign = tf.assign(weights_ori, weights)
                reg = deviation_regularizer(weights - weights_ori)
                tf.add_to_collection(tf.GraphKeys.READY_OP, assign)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            ptr.append(tf.matmul(_input_tensor, weights))
        ptr = tf.add_n(ptr)
        if use_bias:
            name = "bias"
            bias = tf.get_variable(
                name, shape=(output_dim, ), dtype=tf.float32,
                initializer=bias_initializer, trainable=bias_trainable
            )
            if bias_regularizer is not None:
                reg = bias_regularizer(bias)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            if deviation_regularizer is not None:
                name += "_ori"
                bias_ori = tf.get_variable(
                    name, shape=(output_dim, ), dtype=tf.float32,
                    initializer=tf.zeros_initializer(), trainable=False
                )
                assign = tf.assign(bias_ori, bias)
                reg = deviation_regularizer(bias - bias_ori)
                tf.add_to_collection(tf.GraphKeys.READY_OP, assign)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            ptr = tf.add(ptr, bias)
    return ptr


def gan_loss(
        true: tf.Tensor, fake: tf.Tensor, eps: float = 1e-8
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    with tf.name_scope("d_loss"):
        d_loss = tf.negative(tf.reduce_mean(
            tf.log(true + eps) + tf.log(1. - fake + eps)
        ), name="d_loss")
    with tf.name_scope("g_loss"):
        g_loss = tf.negative(tf.reduce_mean(
            tf.log(fake + eps)
        ), name="g_loss")
    return d_loss, g_loss
