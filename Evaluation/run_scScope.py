#! /usr/bin/env python
# by caozj
# Jun 5, 2019
# 3:42:21 PM


import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import Cell_BLAST as cb
import scscope as DeepImpute
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)

    parser.add_argument("--n-latent", dest="n_latent", type=int, default=10)
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=1000)

    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    output_path = os.path.dirname(cmd_args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    return cmd_args


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(
        cmd_args.input, sparsify=True
    ).normalize()
    if cmd_args.clean is not None:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    if cmd_args.genes is not None:
        dataset = dataset[:, dataset.uns[cmd_args.genes]]
    dataset = dataset.exprs.log1p().toarray()
    start_time = time.time()
    model = DeepImpute.train(
        dataset, cmd_args.n_latent,
        max_epoch=cmd_args.n_epochs, random_seed=cmd_args.seed
    )
    latent, _imputed_val, _batch_effect = DeepImpute.predict(dataset, model)
    cb.data.write_hybrid_path(
        time.time() - start_time,
        "//".join([cmd_args.output, "time"])
    )
    cb.data.write_hybrid_path(
        latent,
        "//".join([cmd_args.output, "latent"])
    )


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
