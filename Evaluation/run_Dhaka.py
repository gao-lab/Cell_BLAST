#! /usr/bin/env python
# by caozj
# Jun 5, 2019
# 7:50:50 PM


import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "0"
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from autoencoder import Dhaka
import Cell_BLAST as cb
import utils



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)

    parser.add_argument("--n-latent", dest="n_latent", type=int, default=3)
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=100)
    # Reducing epoch number to the maximum of author recommendation,
    # because Dhaka does not support early stopping and we see
    # numerical instability with larger number of epochs.

    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)  # Not exactly be reproducible though
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    cmd_args.output_path = os.path.dirname(cmd_args.output)
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
    K.set_session(sess)
    return cmd_args


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(
        cmd_args.input, sparsify=True
    ).normalize(target=100000)  # Example data seem to be normalized to 100,000
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    if cmd_args.genes is not None:
        dataset = dataset[:, dataset.uns[cmd_args.genes]]
    dataset = np.log2(dataset.exprs.toarray() + 1)
    mat_file = os.path.join(cmd_args.output_path, "matrix.txt.gz")
    res_file = os.path.join(cmd_args.output_path, "output_datafile")
    np.savetxt(mat_file, dataset)
    start_time = time.time()
    Dhaka.Dhaka(
        mat_file, latent_dim=cmd_args.n_latent,
        N_starts=1, epochs=cmd_args.n_epochs, output_datafile=res_file,
        to_cluster=0, gene_selection=0, to_plot=0, relative_expression=0
    )
    cb.data.write_hybrid_path(
        time.time() - start_time,
        "//".join([cmd_args.output, "time"])
    )
    cb.data.write_hybrid_path(
        np.loadtxt(res_file + ".txt"),
        "//".join([cmd_args.output, "latent"])
    )
    os.remove(mat_file)
    os.remove(res_file + ".txt")


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
