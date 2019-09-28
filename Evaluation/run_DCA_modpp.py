#! /usr/bin/env python
# by caozj
# Jun 4, 2019
# 8:09:11 PM


import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import argparse
import numpy as np
import dca_modpp.api
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)

    parser.add_argument("--n-latent", dest="n_latent", type=int, default=32)
    parser.add_argument("--n-hidden", dest="n_hidden", type=int, default=64)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=1)

    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=1000)
    parser.add_argument("--patience", dest="patience", type=int, default=30)

    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)  # Not exactly be reproducible though
    parser.add_argument("-t", "--threads", dest="threads", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)

    cmd_args = parser.parse_args()
    cmd_args.output_path = os.path.dirname(cmd_args.output)
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    return cmd_args


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.input, sparsify=True)
    if cmd_args.clean is not None:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    if cmd_args.genes is not None:
        genes = dataset.uns[cmd_args.genes]
    else:
        genes = None
    dataset = dataset.to_anndata()
    start_time = time.time()
    dataset, model = dca_modpp.api.dca(
        dataset, genes, mode="latent", normalize_per_cell=10000, scale=False,
        hidden_size=
        (cmd_args.n_hidden, ) * cmd_args.n_layers +
        (cmd_args.n_latent, ) +
        (cmd_args.n_hidden, ) * cmd_args.n_layers,
        epochs=cmd_args.n_epochs, early_stop=cmd_args.patience,
        random_state=cmd_args.seed, threads=cmd_args.threads,
        return_model=True, copy=True
    )
    cb.data.write_hybrid_path(
        time.time() - start_time,
        "//".join([cmd_args.output, "time"])
    )
    cb.data.write_hybrid_path(
        dataset.obsm["X_dca"],
        "//".join([cmd_args.output, "latent"])
    )
    model.encoder.save(os.path.join(cmd_args.output_path, "model.h5"))
    np.savetxt(os.path.join(cmd_args.output_path, "genes.txt"), genes, "%s")


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
