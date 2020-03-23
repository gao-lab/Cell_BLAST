r"""
Abstract module class for DIRECTi
"""


import os
import sys
import typing

import tensorflow as tf

from . import utils


class Module(object):

    r"""
    Build an empty module that does nothing

    Parameters
    ----------
    name
        name of the module
    """

    def __init__(self, name: typing.Optional[str] = None) -> None:
        self.name = name
        self.scope_safe_name = utils.scope_free(name)
        self.vars_to_save = []
        self.on_epoch_end = []

    def _save_weights(self, sess: tf.Session, path: str) -> None:
        if self.vars_to_save:
            if not os.path.exists(path):
                os.makedirs(path)
            tf.train.Saver(
                var_list=self.vars_to_save, max_to_keep=1
            ).save(sess, os.path.join(path, "save.ckpt"), write_meta_graph=False)

    def _load_weights(
            self, sess: tf.Session, path: str, fast: bool = False
    ) -> None:
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
        if failed_vars:
            utils.logger.info("%d variables failed to load.", len(failed_vars))
            utils.logger.debug(str(failed_vars))

    def _compile(self, optimizer: str, lr: float) -> None:
        pass

    @staticmethod
    def _build_feed_dict(*args, **kwargs) -> typing.Mapping:  # pylint: disable=unused-argument
        return {}

    def __bool__(self) -> bool:
        return False

    def _get_config(self) -> typing.Mapping:
        return {
            "class": ".".join((self.__module__, self.__class__.__qualname__)),
            "name": self.name
        }

    @staticmethod
    def _load_config(config: typing.Mapping) -> "Module":
        class_split = config["class"].split(".")
        module_name = ".".join(class_split[:-1])
        class_name = class_split[-1]
        c = getattr(sys.modules[module_name], class_name)
        del config["class"]
        return c(**config)
