"""
Abstract module class for DIRECTi
"""


import sys
import os
import re
import tensorflow as tf
from . import message


class Module(object):

    """
    Build an empty module that does nothing

    Parameters
    ----------
    name : str
        name of the module, by default None
    """

    def __init__(self, name=None):
        self.name = name
        self.scope_safe_name = re.sub("[^A-Za-z0-9_.\\-]", "_", name)
        self.vars_to_save = []
        self.on_epoch_end = []

    def _save_weights(self, sess, path):
        if self.vars_to_save:
            if not os.path.exists(path):
                os.makedirs(path)
            tf.train.Saver(
                var_list=self.vars_to_save, max_to_keep=1
            ).save(sess, os.path.join(path, "save.ckpt"))

    def _load_weights(self, sess, path, verbose=1, fast=False):
        if fast:
            tf.train.Saver(
                var_list=self.vars_to_save, max_to_keep=1
            ).restore(sess, os.path.join(path, "save.ckpt"))
            return
        failed_vars = []
        for var_to_save in self.vars_to_save:
            try:
                tf.train.Saver(
                    var_list=[var_to_save], max_to_keep=1
                ).restore(sess, os.path.join(path, "save.ckpt"))
            except Exception:
                failed_vars.append(var_to_save)
        if failed_vars and verbose:
            message.warning("%d variables failed to load!" % len(failed_vars))
            if verbose > 1:
                print(failed_vars)

    def _compile(self, optimizer, lr):
        pass

    @staticmethod
    def _build_feed_dict(*args, **kwargs):
        return dict()

    def __bool__(self):
        return False

    def _get_config(self):
        return {
            "class": ".".join((self.__module__, self.__class__.__qualname__)),
            "name": self.name
        }

    @staticmethod
    def _load_config(config):
        class_split = config["class"].split(".")
        module_name = ".".join(class_split[:-1])
        class_name = class_split[-1]
        c = getattr(sys.modules[module_name], class_name)
        del config["class"]
        return c(**config)
