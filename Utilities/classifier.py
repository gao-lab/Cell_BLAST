import numpy as np
import tensorflow as tf
from . import model
from . import nn
from . import utils


class Classifier(model.Model):

    def _init_graph(self, x_dim, latent_dim, n_class,
                    h_dim=512, depth=1, dropout=0.):

        with tf.name_scope("placeholder/"):
            self.x = tf.placeholder(
                dtype=tf.float32, shape=(None, x_dim), name="x")
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(None, n_class), name="y")
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag")

        ptr = nn.mlp(
            self.x, [h_dim] * depth, dropouts=dropout,
            batch_normalizations=True, training_flag=self.training_flag
        )
        self.latent = tf.identity(
            nn.dense(ptr, latent_dim, scope="latent"), name="latent")
        self.pred = tf.identity(
            nn.dense(self.latent, n_class, scope="pred"), name="pred")

        self.loss = tf.losses.softmax_cross_entropy(self.y, self.pred)
        self.vars_to_save += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def _compile(self, optimizer=tf.train.RMSPropOptimizer, lr=1e-3):
        with tf.variable_scope("optimizer"):
            self.step = optimizer(lr).minimize(self.loss)

    def _fit_epoch(self, data_dict, batch_size=128):
        loss = 0.

        @utils.minibatch(batch_size, desc="training", use_last=False)
        def _train(data_dict):
            nonlocal loss
            feed_dict = {
                self.x: utils.densify(data_dict["x"]),
                self.y: utils.densify(data_dict["y"]),
                self.training_flag: True
            }
            _, l = self.sess.run([self.step, self.loss], feed_dict=feed_dict)
            loss += l * data_dict.size

        _train(data_dict)
        loss /= data_dict.size
        print("train=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[tf.Summary.Value(
            tag="train_loss", simple_value=loss
        )])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(self, data_dict, batch_size=128):
        loss = 0.

        @utils.minibatch(batch_size, desc="validation", use_last=True)
        def _validate(data_dict):
            nonlocal loss
            feed_dict = {
                self.x: utils.densify(data_dict["x"]),
                self.y: utils.densify(data_dict["y"]),
                self.training_flag: False
            }
            l = self.sess.run(self.loss, feed_dict=feed_dict)
            loss += l * data_dict.size

        _validate(data_dict)
        loss /= data_dict.size
        print("val=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[tf.Summary.Value(
            tag="val_loss", simple_value=loss
        )])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return self.sess.graph.get_tensor_by_name(key + ":0")

    def fetch(self, tensor, x, batch_size=128):
        tensor_shape = tuple(
            item for item in tensor.get_shape().as_list() if item is not None)
        result = np.empty((x.shape[0],) + tuple(tensor_shape))

        @utils.minibatch(batch_size, desc="fetch", use_last=True)
        def _fetch(x, result):
            feed_dict = {
                self.x: utils.densify(x),
                self.training_flag: False
            }
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)
        _fetch(x, result)
        return result

    def classify(self, x, batch_size=128):
        return self.fetch(tf.nn.softmax(self.pred), x, batch_size=batch_size)

    def inference(self, x, batch_size=128):
        return self.fetch(self.latent, x, batch_size=batch_size)
