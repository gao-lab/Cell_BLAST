"""
Latent space / encoder modules for DIRECTi
"""


import numpy as np
import tensorflow as tf
import sklearn.metrics
from . import nn
from . import module
from . import utils


class Latent(module.Module):
    """
    Parent class for latent variable modules.
    """
    def __init__(self, latent_dim, h_dim=128, depth=1, dropout=0.0,
                 lambda_reg=0.0, fine_tune=False, deviation_reg=0.0, name="Latent"):
        super(Latent, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.depth = depth
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.fine_tune = fine_tune
        self.deviation_reg = deviation_reg
        self.deviation_regularizer = \
            (lambda x: self.deviation_reg * tf.reduce_mean(tf.square(x))) \
            if self.fine_tune and self.deviation_reg > 0 else None

    def _build_latent(self, pre_latent, training_flag, scope="encoder"):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `build_latent` from `Latent`!")

    def _build_regularizer(self, training_flag, epoch, scope="regularizer"):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `build_regularizer` from `Latent`!")

    def __bool__(self):
        return True

    def _get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "h_dim": self.h_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "fine_tune": self.fine_tune,
            "deviation_reg": self.deviation_reg,
            **super(Latent, self)._get_config()
        }


class Gau(Latent):
    """
    Build a Gaussian latent variable module.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent variable.
    h_dim : int
        Dimensionality of hidden layers in MLP, by default 128.
    depth : int
        Number of hidden layers in MLP, by default 1.
    dropout : float
        Dropout rate, by default 0.0.
    lambda_reg : float
        Regularization strength on the latent variable, by default 0.001.
    name : str
        Name of the module, by default "Gau".
    """
    def __init__(self, latent_dim, h_dim=128, depth=1, dropout=0.0,
                 lambda_reg=0.001, fine_tune=False, deviation_reg=0.0, name="Gau"):
        super(Gau, self).__init__(latent_dim, h_dim, depth, dropout, lambda_reg,
                                  fine_tune, deviation_reg, name)

    def _build_latent(self, x, training_flag, scope="encoder"):
        self.build_latent_scope = "%s/%s" % (scope, self.scope_safe_name)
        with tf.variable_scope(self.build_latent_scope):
            dense_kwargs = [dict(
                deviation_regularizer=self.deviation_regularizer
            )] * self.depth
            dense_kwargs[0]["weights_trainable"] = not self.fine_tune
            ptr = nn.mlp(
                x, [self.h_dim] * self.depth,
                dropout=self.dropout, batch_normalization=True,
                dense_kwargs=dense_kwargs, training_flag=training_flag
            )
            self.gau = tf.identity(nn.dense(
                ptr, self.latent_dim,
                deviation_regularizer=self.deviation_regularizer
            ), name="gau")
        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_latent_scope)
        return self.gau

    def _build_regularizer(self, training_flag, epoch, scope="discriminator"):
        self.gaup_sampler = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.build_regularizer_scope = "%s/%s" % (scope, self.scope_safe_name)
        with tf.variable_scope(self.build_regularizer_scope, reuse=tf.AUTO_REUSE):
            self.gaup = self.gaup_sampler.sample((
                tf.shape(self.gau)[0], self.latent_dim))
            gau_pred = tf.sigmoid(nn.dense(nn.mlp(
                self.gau, [self.h_dim] * self.depth,
                dropout=self.dropout, training_flag=training_flag
            ), 1), name="pred")
            gaup_pred = tf.sigmoid(nn.dense(nn.mlp(
                self.gaup, [self.h_dim] * self.depth,
                dropout=self.dropout, training_flag=training_flag
            ), 1), name="prior_pred")
            self.gau_d_loss, self.gau_g_loss = nn.gan_loss(gaup_pred, gau_pred)

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_d_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_g_loss)

        return self.lambda_reg * self.gau_g_loss

    def _compile(self, optimizer, lr):
        with tf.variable_scope("optimize/%s" % self.scope_safe_name):
            optimizer = tf.train.__dict__[optimizer]
            self.step = optimizer(lr).minimize(
                self.lambda_reg * self.gau_d_loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    self.build_regularizer_scope
                )
            )
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.step)


class CatGau(Latent):
    """
    Build a double latent variable module, with a continuous Gaussian
    latent variable and a one-hot categorical latent variable for intrainsic
    clustering of the data. These two are then combined into a single embedding
    vector.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent variable.
    cat_dim : int
        Number of intrinsic clusters.
    h_dim : int
        Dimensionality of hidden layers in MLP, by default 128.
    depth : int
        Number of hidden layers in MLP, by default 1.
    dropout : float
        Dropout rate, by default 0.0.
    multiclass_adversarial : bool
        Whether to use multi-class adversarial regularization on the
        Gaussian latent variable, by default False.
        Setting this to True makes each intrinsic cluster more Gaussian-like.
    cat_merge : bool
        Whether to enable heuristic cluster merging during training,
        by default False.
    min_silhouette : float
        Minimal average silhouette score required to prevent an instrinsic
        cluster from being merged, by default 0.0.
    patience : int
        Execute heuristic cluster merging under a "fast-ring" early stop
        mechanism, with early stop patience specified by this argument,
        by default 10.
    lambda_reg : float
        Regularization strength on the latent variables, by default 0.001.
    name : str
        Name of the module, by default "CatGau".
    """
    def __init__(self, latent_dim, cat_dim, h_dim=128, depth=1, dropout=0.0,
                 multiclass_adversarial=False,
                 cat_merge=False, min_silhouette=0.0, patience=10,
                 lambda_reg=0.001, fine_tune=False, deviation_reg=0.0,
                 name="CatGau"):
        super(CatGau, self).__init__(latent_dim, h_dim, depth, dropout,
                                     lambda_reg, fine_tune, deviation_reg, name)
        self.cat_dim = cat_dim
        self.multiclass_adversarial = multiclass_adversarial
        self.min_silhouette = min_silhouette
        self.patience = patience
        self.cat_merge = cat_merge
        if cat_merge:
            self.on_epoch_end.append(self._cat_merge)

    def _build_latent(self, x, training_flag, scope="encoder"):
        self.build_latent_scope = "%s/%s" % (scope, self.scope_safe_name)
        with tf.variable_scope(self.build_latent_scope):
            dense_kwargs = [dict(
                deviation_regularizer=self.deviation_regularizer
            )] * self.depth
            dense_kwargs[0]["weights_trainable"] = not self.fine_tune
            ptr = nn.mlp(
                x, [self.h_dim] * self.depth,
                dropout=self.dropout, batch_normalization=True,
                dense_kwargs=dense_kwargs,
                training_flag=training_flag
            )
            with tf.variable_scope("cat"):
                self.cat_logit = tf.identity(nn.dense(
                    ptr, self.cat_dim,
                    deviation_regularizer=self.deviation_regularizer
                ), name="cat_logit")
                self.cat = tf.nn.softmax(self.cat_logit, name="cat")
            with tf.variable_scope("gau"):
                self.gau = tf.identity(nn.dense(
                    ptr, self.latent_dim,
                    deviation_regularizer=self.deviation_regularizer
                ), name="gau")
            self.latent = self.gau + nn.dense(
                self.cat, self.latent_dim, use_bias=False,
                weights_initializer=tf.random_normal_initializer(stddev=0.1),
                deviation_regularizer=self.deviation_regularizer,
                scope="cluster_head"
            )
        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_latent_scope)
        return self.latent

    def _build_regularizer(self, training_flag, epoch, scope="discriminator"):
        self.catp_mask = tf.get_variable(
            "catp_mask", initializer=np.ones(self.cat_dim), trainable=False)
        self.vars_to_save.append(self.catp_mask)
        self.catp_sampler = tf.distributions.Categorical(
            probs=self.catp_mask / tf.reduce_sum(self.catp_mask))
        self.gaup_sampler = tf.distributions.Normal(loc=0.0, scale=1.0)
        if self.multiclass_adversarial:
            return self._build_multiclass_regularizer(training_flag, scope)
        return self._build_binary_regularizer(training_flag, scope)

    def _build_binary_regularizer(self, training_flag, scope="discriminator"):
        self.build_regularizer_scope = "%s/%s" % (scope, self.scope_safe_name)
        with tf.variable_scope(self.build_regularizer_scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("cat"):
                self.catp = tf.one_hot(
                    self.catp_sampler.sample(tf.shape(self.cat)[0]),
                    depth=self.cat_dim
                )
                cat_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.cat, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="pred")
                catp_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.catp, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="prior_pred")
                self.cat_d_loss, self.cat_g_loss = \
                    nn.gan_loss(catp_pred, cat_pred)
            with tf.variable_scope("gau"):
                self.gaup = self.gaup_sampler.sample((
                    tf.shape(self.gau)[0], self.latent_dim))
                gau_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.gau, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="pred")
                gaup_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.gaup, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="prior_pred")
                self.gau_d_loss, self.gau_g_loss = \
                    nn.gan_loss(gaup_pred, gau_pred)

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.cat_d_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.cat_g_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_d_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_g_loss)
        return self.lambda_reg * (self.cat_g_loss + self.gau_g_loss)

    def _build_multiclass_regularizer(self, training_flag, scope="discriminator"):
        self.build_regularizer_scope = "%s/%s" % (scope, self.scope_safe_name)
        with tf.variable_scope(self.build_regularizer_scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("cat"):
                self.catp = tf.one_hot(
                    self.catp_sampler.sample(tf.shape(self.cat)[0]),
                    depth=self.cat_dim
                )
                cat_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.cat, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="pred")
                catp_pred = tf.sigmoid(nn.dense(nn.mlp(
                    self.catp, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), 1), name="prior_pred")
                self.cat_d_loss, self.cat_g_loss = \
                    nn.gan_loss(catp_pred, cat_pred)
            with tf.variable_scope("gau"):
                self.gaup = self.gaup_sampler.sample((
                    tf.shape(self.gau)[0], self.latent_dim))
                gau_logits = nn.dense(nn.mlp(
                    self.gau, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), self.cat_dim + 1)
                gaup_logits = nn.dense(nn.mlp(
                    self.gaup, [self.h_dim] * self.depth,
                    dropout=self.dropout, training_flag=training_flag
                ), self.cat_dim + 1)
                true = tf.concat([
                    tf.concat([
                        self.cat,
                        tf.zeros((tf.shape(self.cat)[0], 1))
                    ], axis=1),
                    tf.concat([
                        tf.zeros((tf.shape(gaup_logits)[0], self.cat_dim)),
                        tf.ones((tf.shape(gaup_logits)[0], 1))
                    ], axis=1)
                ], axis=0)
                logits = tf.concat([gau_logits, gaup_logits], axis=0)
                self.gau_d_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.stop_gradient(true), logits=logits,
                    ), name="d_loss"
                )
                self.gau_g_loss = tf.negative(self.gau_d_loss, name="g_loss")

        self.vars_to_save += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.build_regularizer_scope)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.cat_d_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.cat_g_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_d_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, self.gau_g_loss)
        return self.lambda_reg * (self.cat_g_loss + self.gau_g_loss)

    def _compile(self, optimizer, lr):
        with tf.variable_scope("optimize/%s" % self.scope_safe_name):
            optimizer = tf.train.__dict__[optimizer]
            self.step = optimizer(lr).minimize(
                self.lambda_reg * (self.cat_d_loss + self.gau_d_loss),
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    self.build_regularizer_scope
                )
            )
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.step)

    # On epoch end heuristic
    def _cat_merge(self, model, train_data_dict, val_data_dict, loss):

        # Initialization
        if "_cat_merge_dict" not in dir(self):
            self._cat_merge_dict = {
                "record": np.inf,
                "countdown": self.patience,
                "converged": False
            }
        d = self._cat_merge_dict

        # Guard operatability
        if model.sess.run(self.catp_mask).sum() == 1:
            d["converged"] = True
            return d["converged"]

        # Guard entrance
        if loss < d["record"]:
            d["record"] = loss
            d["countdown"] = self.patience
        else:
            d["countdown"] -= 1

        # Entrance
        if d["countdown"] == 0:
            removed_clusters = set(np.where(
                model.sess.run(self.catp_mask) == 0
            )[0])

            # Identify cluster heads that are not assigned any samples
            cluster = model._fetch(
                self.cat, train_data_dict["exprs"]
            ).argmax(axis=1).astype(np.int)
            population = np.eye(self.cat_dim)[cluster, :].sum(axis=0)
            remove_idx = set(np.where(
                population <= 1
            )[0]).difference(removed_clusters)
            reason = "emptyness"

            # Identify clusters that are not clearly separated
            if not remove_idx:
                latent = model.inference(train_data_dict["exprs"])
                if train_data_dict.size > 10000:
                    subsample_idx = model.random_state.choice(
                        train_data_dict.size, 10000, replace=False)
                    latent = latent[subsample_idx]
                    cluster = cluster[subsample_idx]
                    population = np.eye(self.cat_dim)[cluster, :].sum(axis=0)
                sample_silhouette = sklearn.metrics.silhouette_samples(
                    latent, cluster)
                cluster_silhouette = np.empty(len(population))
                for i in range(len(population)):
                    cluster_silhouette[i] = sample_silhouette[
                        cluster == i
                    ].mean() if population[i] > 0 else np.inf
                population[cluster_silhouette > self.min_silhouette] = np.inf
                remove_candidate = set(
                    np.where(np.isfinite(population))[0]
                ).difference(removed_clusters).difference(self._safe_cat(model))
                if remove_candidate:
                    remove_candidate = list(remove_candidate)
                    remove_idx = {remove_candidate[
                        population[remove_candidate].argmin()
                    ]}
                    reason = "inconfidence"
                else:
                    remove_idx = dict()

            # Merge identified clusters
            if remove_idx:
                d["converged"] = False
                current_value = model.sess.run(self.catp_mask)
                current_value[list(remove_idx)] = 0
                model.sess.run(tf.assign(self.catp_mask, current_value))
                print("Cluster %s removed because of %s." % (
                    ",".join([str(item) for item in remove_idx]), reason
                ))
            else:
                d["converged"] = True
                print("Nothing done in cluster manipulation.")

            d["countdown"] = self.patience
            d["record"] = np.inf

        return d["converged"]

    def _safe_cat(self, model):  # define which clusters should safe from merging
        return []

    def _get_config(self):
        return {
            "cat_dim": self.cat_dim,
            "multiclass_adversarial": self.multiclass_adversarial,
            "cat_merge": self.cat_merge,
            "min_silhouette": self.min_silhouette,
            "patience": self.patience,
            **super(CatGau, self)._get_config()
        }


class SemiSupervisedCatGau(CatGau):
    """
    Build a double latent variable module, with a continuous Gaussian
    latent variable and a one-hot categorical latent variable for intrinsic
    clustering of the data. The categorical latent supports semi-supervision.
    The two latents are then combined into a single embedding vector.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent variable.
    cat_dim : int
        Number of intrinsic clusters.
    h_dim : int
        Dimensionality of hidden layers in MLP, by default 128.
    depth : int
        Number of hidden layers in MLP, by default 1.
    dropout : float
        Dropout rate, by default 0.0.
    multiclass_adversarial : bool
        Whether to use multi-class adversarial regularization on the
        Gaussian latent variable, by default False.
        Setting this to True makes each intrinsic cluster more Gaussian-like.
    cat_merge : bool
        Whether to enable heuristic cluster merging during training,
        by default False.
    min_silhouette : float
        Minimal average silhouette score required to prevent an instrinsic
        cluster from being merged, by default 0.0.
    patience : int
        Execute heuristic cluster merging under a "fast-ring" early stop
        mechanism, with early stop patience specified by this argument,
        by default 10.
    lambda_sup : float
        Supervision strength, by default 10.0.
    background_catp : float
        Unnormalized background prior distribution of the intrinsic
        clustering latent, by default 1e-3.
        For each supervised cell in a minibatch,  unnormalized prior
        probability of the corresponding cluster will increase by 1,
        so this parameter determines how much to trust supervision class
        frequency. Additionally it also balances between supervision and
        identifying new clusters.
    lambda_reg : float
        Regularization strength on the latent variables, by default 0.001.
    name : str
        Name of latent module, by default "SemiSupervisedCatGau".
    """
    def __init__(self, latent_dim, cat_dim, h_dim=128, depth=1, dropout=0.0,
                 multiclass_adversarial=False,
                 cat_merge=False, min_silhouette=0.0, patience=10,
                 lambda_sup=10.0, background_catp=1e-3,
                 lambda_reg=0.001, fine_tune=False, deviation_reg=0.0,
                 name="SemiSupervisedCatGau"):
        super(SemiSupervisedCatGau, self).__init__(
            latent_dim, cat_dim, h_dim, depth, dropout, multiclass_adversarial,
            cat_merge, min_silhouette, patience, lambda_reg,
            fine_tune, deviation_reg, name
        )
        self.lambda_sup = lambda_sup
        self.background_catp = background_catp

    def _build_regularizer(self, training_flag, epoch, scope="discriminator"):
        with tf.name_scope("placeholder/"):
            self.cats = tf.placeholder(
                dtype=tf.float32, shape=(None, self.cat_dim), name="cats")
        self.cats_coverage = tf.get_variable(
            "cats_coverage", initializer=np.zeros(
                self.cat_dim, dtype=np.bool_
            ), trainable=False
        )  # Used in cat manipulation to avoid merging supervised label
        self.vars_to_save.append(self.cats_coverage)

        self.catp_mask = tf.get_variable(
            "catp_mask", initializer=np.ones(
                self.cat_dim, dtype=np.float32
            ) * self.background_catp, trainable=False
        )
        self.catp_prob = self.catp_mask + tf.reduce_sum(self.cats, axis=0)
        self.catp_sampler = tf.distributions.Categorical(
            probs=self.catp_prob / tf.reduce_sum(self.catp_prob))
        self.gaup_sampler = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.vars_to_save.append(self.catp_mask)

        with tf.name_scope("semi_supervision/%s" % self.scope_safe_name):
            mask = tf.cast(tf.reduce_sum(self.cats, axis=1) > 0, tf.int32)
            masked_cat_logit = tf.dynamic_partition(self.cat_logit, mask, 2)[1]
            masked_cats = tf.dynamic_partition(self.cats, mask, 2)[1]
            self.sup_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=masked_cats, logits=masked_cat_logit
                ), name="supervised_loss"
            )
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.sup_loss)

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(self.cats_coverage, tf.logical_or(
                self.cats_coverage,
                tf.cast(tf.reduce_max(self.cats, axis=0), tf.bool)
            ))
        )
        return self.lambda_sup * self.sup_loss + (
            self._build_multiclass_regularizer
            if self.multiclass_adversarial
            else self._build_binary_regularizer
        )(training_flag, scope)

    def _build_feed_dict(self, data_dict):
        return {self.cats: utils.densify(data_dict[self.name])}

    def _safe_cat(self, model):
        return np.where(model.sess.run(self.cats_coverage))[0]

    def _get_config(self):
        return {
            "lambda_sup": self.lambda_sup,
            "background_catp": self.background_catp,
            **super(SemiSupervisedCatGau, self)._get_config()
        }
