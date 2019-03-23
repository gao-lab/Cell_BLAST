"""
Model initialization, training and saving/loading framework used by DIRECTi
"""


from builtins import input
import os
import time
import json
import traceback

import numpy as np
import tensorflow as tf

from . import utils
from . import message


class Model(object):
    """
    Abstract model class, providing a framework for model initialization,
    training, saving and loading.
    """
    def __init__(self, random_seed=None, path=".", **kwargs):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.random_state = np.random.RandomState(random_seed)
        self.graph = tf.Graph()
        self.vars_to_save = []
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self._init_graph(**kwargs)
            self._init_session()

    @utils.with_self_graph
    def compile(self, optimizer, lr, initialize_weights=True):
        """
        Compile the model and get ready for fitting.

        Parameters
        ----------
        optimizer : str
            Name of the optimizer to use.
        lr : float
            Learning rate.
        initialize_weights : bool
            Whether to initialize model weights, by default True.

        Returns
        -------
        model : Cell_BLAST.model.Model
            A compiled model.
        """
        self._compile(optimizer, lr)
        self._summarize()
        if initialize_weights:
            self.sess.run(tf.global_variables_initializer())
        return self

    # Called with self.graph.as_default
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    # Called with self.graph.as_default
    def _init_graph(self, **kwargs):
        with tf.variable_scope("epoch", reuse=tf.AUTO_REUSE):
            self.epoch = tf.get_variable(
                "epoch", shape=(), dtype=tf.int32, trainable=False)

    # Called with self.graph.as_default
    def _compile(self, optimizer, lr):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `_compile` from `Model`!")

    # Called with self.graph.as_default
    def _summarize(self):
        self.summarizer = tf.summary.FileWriter(
            os.path.join(self.path, "summary"),
            graph=self.graph, flush_secs=10
        )

    def close(self):
        """
        Clean up and close the model.
        """
        self.sess.close()
        # tf.reset_default_graph()

    @utils.with_self_graph
    def fit(self, data_dict, val_split=0.1, epoch=100,
            patience=np.inf, on_epoch_end=None, progress_bar=True, **kwargs):
        """
        This function wraps an epoch-by-epoch update function into
        complete training process that supports data splitting, shuffling
        and early stop.
        """
        if on_epoch_end is None:
            on_epoch_end = []
        data_dict = utils.DataDict(data_dict)

        # Leave out validation set
        data_dict = data_dict.shuffle(self.random_state)
        data_size = data_dict.size
        train_data_dict = data_dict[int(val_split * data_size):]
        val_data_dict = data_dict[:int(val_split * data_size)]

        # Fit preparation
        loss_record = np.inf
        patience_countdown = patience
        saver = tf.train.Saver(max_to_keep=1)
        ckpt_file = os.path.join(self.path, "checkpoint")

        # Fit loop
        for epoch_idx in range(epoch):
            self.epoch_report = ""
            try:
                self.epoch_report += "[%s epoch %d] " % (
                    self.__class__.__name__, epoch_idx)
                self.sess.run(self.epoch.assign(epoch_idx))

                try:
                    t_start = time.time()
                    self._fit_epoch(
                        train_data_dict.shuffle(self.random_state),
                        progress_bar=progress_bar, **kwargs
                    )
                    loss = self._val_epoch(
                        val_data_dict,
                        progress_bar=progress_bar, **kwargs
                    )
                    self.epoch_report += "time elapsed=%.1fs" % (
                        time.time() - t_start)
                except Exception:  # pragma: no cover
                    print("\n==== Oops! Model has crashed... ====\n")
                    traceback.print_exc()
                    print("\n====================================\n")
                    break

                all_converged = True
                for fn in on_epoch_end:
                    all_converged = fn(
                        self, train_data_dict, val_data_dict, loss
                    ) and all_converged

                # Early stop
                if all_converged:
                    if np.isfinite(patience) and loss < loss_record:
                        self.epoch_report += " Best save..."
                        latest_checkpoint = saver.save(
                            self.sess, ckpt_file, global_step=epoch_idx)
                        patience_countdown = patience
                        loss_record = loss
                    else:
                        patience_countdown -= 1
                    print(self.epoch_report)
                    if patience_countdown == 0:
                        break
                else:
                    loss_record = np.inf  # In case all_converged is exited
                    patience_countdown = patience
                    if epoch_idx % 10 == 0:  # Save regularly
                        self.epoch_report += " Regular save..."
                        latest_checkpoint = saver.save(
                            self.sess, ckpt_file, global_step=epoch_idx)
                    print(self.epoch_report)

            except KeyboardInterrupt:  # pragma: no cover
                print("\n\n==== Caught keyboard interruption! ====\n")
                success_flag = False
                break_flag = False
                while not success_flag:
                    choice = input("Stop model training? (y/n) ")
                    if choice == "y":
                        break_flag = True
                        break
                    elif choice == "n":
                        success_flag = True
                if break_flag:
                    break

        # Fit finish
        if "latest_checkpoint" in locals():
            print("Restoring best model...")
            saver.restore(self.sess, latest_checkpoint)
        return self

    # Called with self.graph.as_default
    def _fit_epoch(self, data_dict, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `_fit_epoch` from `Model`!")

    # Called with self.graph.as_default
    def _val_epoch(self, data_dict, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "Calling virtual `_val_epoch` from `Model`!")

    def _get_config(self):
        return {}

    def _save_config(self, file):
        with open(file, "w") as f:
            json.dump(self._get_config(), f, indent=4)

    @classmethod
    def _load_config(cls, file, **kwargs):
        with open(file, "r") as f:
            config = json.load(f)
        return cls(path=os.path.dirname(file), **config, **kwargs)

    @utils.with_self_graph
    def _save_weights(self, path):
        if self.vars_to_save:
            if not os.path.exists(path):
                os.makedirs(path)
            tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1).save(
                self.sess, os.path.join(path, "save.ckpt"))

    @utils.with_self_graph
    def _load_weights(self, path, verbose=1):
        failed_vars = []
        for var_to_save in self.vars_to_save:
            try:
                tf.train.Saver(
                    var_list=[var_to_save], max_to_keep=1
                ).restore(
                    self.sess, os.path.join(path, "save.ckpt")
                )
            except Exception:
                failed_vars.append(var_to_save)
        if failed_vars and verbose:
            message.warning("%d variables failed to load!" % len(failed_vars))
            if verbose > 1:
                print(failed_vars)

    def save(self, path=None, config="config.json", weights="weights"):
        """
        Save model configuration and weights.

        Parameters
        ----------
        path : str
            Path to save the model, by default None. If not given, will use
            ``model.path``.
        config : str
            File to store model configuration (in json format, under ``path``),
            by default "config.json".
        weights : str
            Directory to store model weights (under ``path``), by default
            "weights".
        """
        if path is None:
            path = self.path
        elif not os.path.exists(path):
            os.makedirs(path)
        self._save_config(os.path.join(path, config))
        self._save_weights(os.path.join(path, weights))

    @classmethod
    def load(cls, path, config="config.json", weights="weights",
             verbose=1, **kwargs):
        """
        Load model configuration and weights.

        Parameters
        ----------
        path : str
            Model path.
        config : str
            File that stores model configuration (in json format,
            under ``path``), by default "config.json".
        weights : str
            Directory that stores model weights (under ``path``),
            by default "weights".
        verbose : int
            Verbose level, by default 1.
            If verbose = 0, no information will be printed.
            If verbose = 1, print number of variables that failed to load.
            If verbose = 2, additionally print variable list that failed to load.
        **kwargs
            Additional keyword arguments to be passed to the class constructor

        Returns
        -------
        loaded_model : Cell_BLAST.model.Model
            Loaded model.
        """
        model = cls._load_config(os.path.join(path, config), **kwargs)
        model._load_weights(os.path.join(path, weights), verbose)
        return model
