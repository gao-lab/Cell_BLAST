"""
Neural network operations
"""


import tensorflow as tf


def mlp(input_tensor, hidden_dim, activation=tf.nn.leaky_relu,
        dropout=0.0, batch_normalization=False, dense_kwargs=None,
        training_flag=None, scope="mlp"):

    if not isinstance(hidden_dim, (list, tuple)):
        hidden_dim = [hidden_dim]
    if isinstance(activation, (list, tuple)):
        assert len(activation) == len(hidden_dim)
    else:
        activation = [activation] * len(hidden_dim)
    if isinstance(dropout, (list, tuple)):
        assert len(dropout) == len(hidden_dim)
    else:
        dropout = [dropout] * len(hidden_dim)
    if isinstance(batch_normalization, (list, tuple)):
        assert len(batch_normalization) == len(hidden_dim)
    else:
        batch_normalization = [batch_normalization] * len(hidden_dim)
    if isinstance(dense_kwargs, (list, tuple)):
        assert len(dense_kwargs) == len(hidden_dim)
    else:
        dense_kwargs = [dense_kwargs] * len(hidden_dim)

    with tf.variable_scope(scope):
        ptr = input_tensor
        for l, (_hidden_dim, _activation, _dropout, _batch_normalization, _dense_kwargs) in enumerate(zip(
            hidden_dim, activation, dropout, batch_normalization, dense_kwargs
        )):
            assert not (_dropout or _batch_normalization) or training_flag is not None
            with tf.variable_scope("layer_%d" % l):
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
                ptr = tf.layers.dropout(
                    ptr, rate=_dropout, training=training_flag
                ) if _dropout else ptr
    return ptr


def dense(input_tensor, output_dim, use_bias=True,
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          bias_initializer=tf.zeros_initializer(),
          weights_trainable=True, bias_trainable=True,
          weights_regularizer=None, bias_regularizer=None,
          deviation_regularizer=None, scope="dense"):
    # `input_tensor` can accept a list of tensors instead of concatenating
    # them before passing to `dense`, which enables reusing part of the weight
    # matrix later.
    # Similarly, `weights_initializer`, `weights_trainable` and
    # `weights_regularizer` can also be a list, each element corresponding to
    # weights of an input tensor.
    with tf.variable_scope(scope):
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]
        if isinstance(weights_initializer, (list, tuple)):
            assert len(weights_initializer) == len(input_tensor)
        else:
            weights_initializer = [weights_initializer] * len(input_tensor)
        if isinstance(weights_trainable, (list, tuple)):
            assert len(weights_trainable) == len(input_tensor)
        else:
            weights_trainable = [weights_trainable] * len(input_tensor)
        if isinstance(weights_regularizer, (list, tuple)):
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
            name = "weights_%d" % i if i > 0 else "weights"
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
                tf.add_to_collection(tf.GraphKeys.READY_OP, assign)  # TODO: side effect using READY_OP?
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
                tf.add_to_collection(tf.GraphKeys.READY_OP, assign)  # TODO: side effect using READY_OP?
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
            ptr = tf.add(ptr, bias)
    return ptr


def gan_loss(true, fake, eps=1e-8):
    with tf.name_scope("d_loss"):
        d_loss = tf.negative(tf.reduce_mean(
            tf.log(true + eps) + tf.log(1. - fake + eps)
        ), name="d_loss")
    with tf.name_scope("g_loss"):
        g_loss = tf.negative(tf.reduce_mean(
            tf.log(fake + eps)
        ), name="g_loss")
    return d_loss, g_loss
