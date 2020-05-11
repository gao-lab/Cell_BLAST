#! /usr/bin/env python
# by caozj
# May 1, 2020
# 12:43:26 PM


import os
import argparse
import time
import subprocess
import numpy as np
import pandas as pd

import SAUCIE
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
    parser.add_argument("-b", "--batch-effect", dest="batch_effect", type=str, default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)

    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
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
    dataset = dataset.normalize()
    dataset = dataset[:, dataset.uns[cmd_args.genes]]
    dataset.exprs = np.log1p(dataset.exprs)
    if cmd_args.batch_effect is not None:
        batches = np.unique(dataset.obs[cmd_args.batch_effect])
        for batch in batches:
            dataset[dataset.obs[cmd_args.batch_effect] == batch, :].write_table(
                os.path.join(cmd_args.output_path, "input", f"{batch}.csv"),
                index=False
            )
    else:
        dataset.write_table(
            os.path.join(cmd_args.output_path, "input", "data.csv"),
            index=False
        )

    call_args = [
        "python", os.path.join(SAUCIE.__path__[0], "SAUCIE.py"),
        "--input_dir", os.path.join(cmd_args.output_path, "input"),
        "--output_dir", os.path.join(cmd_args.output_path, "output"),
        "--seed", str(cmd_args.seed),
        "--cluster"
    ]
    if cmd_args.batch_effect is not None:
        call_args.append("--batch_correct")
    start_time = time.time()
    print(f"Running command: {' '.join(call_args)}")
    subprocess.check_call(call_args)
    cb.data.write_hybrid_path(
        time.time() - start_time,
        "//".join([cmd_args.output, "time"])
    )

    if cmd_args.batch_effect is not None:
        latent = np.empty((dataset.shape[0], 2))
        for batch in batches:
            idx = np.where(dataset.obs[cmd_args.batch_effect] == batch)[0]
            latent[idx, :] = pd.read_csv(os.path.join(
                cmd_args.output_path, "output", "clustered", f"{batch}.csv"
            )).loc[:, ["Embedding_SAUCIE1", "Embedding_SAUCIE2"]].to_numpy()
    else:
        latent = pd.read_csv(os.path.join(
            cmd_args.output_path, "output", "clustered", "data.csv"
        )).loc[:, ["Embedding_SAUCIE1", "Embedding_SAUCIE2"]].to_numpy()
    cb.data.write_hybrid_path(latent, "//".join([cmd_args.output, "latent"]))


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
