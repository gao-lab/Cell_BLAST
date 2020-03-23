#! /usr/bin/env python
# by caozj
# Feb 1, 2020
# 12:37:58 PM


import os
import argparse
import time

import numpy as np

import Cell_BLAST as cb
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-m", "--model", dest="model", type=str, required=True)
    parser.add_argument("-b", "--batch-effect", dest="batch_effect", type=str, required=True)
    parser.add_argument("-t", "--target", dest="target", type=str, choices=["zeros", "first", "ones"], default="zeros")
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    os.makedirs(os.path.dirname(cmd_args.output), exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    return cmd_args


def main(cmd_args):
    cb.utils.logger.info("Reading data...")
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.input)
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    model = cb.directi.DIRECTi.load(cmd_args.model)
    data_dict = {
        "exprs": dataset[:, model.genes].exprs,
        "library_size": np.array(dataset.exprs.sum(axis=1)).reshape((-1, 1))
    }
    start_time = time.time()
    if cmd_args.target == "zeros":
        data_dict[cmd_args.batch_effect] = np.zeros((
            dataset.shape[0],
            np.unique(dataset.obs[cmd_args.batch_effect]).size
        ))
    elif cmd_args.target == "first":
        data_dict[cmd_args.batch_effect] = cb.utils.encode_onehot(
            dataset.obs["dataset_name"].astype(object).fillna("IgNoRe"),
            sort=True, ignore="IgNoRe"
        ).toarray()
        data_dict[cmd_args.batch_effect][:, 0] = 1.0
        data_dict[cmd_args.batch_effect][:, 1:] = 0.0
    else:  # cmd_args.target == "ones":
        data_dict[cmd_args.batch_effect] = np.ones((
            dataset.shape[0],
            np.unique(dataset.obs[cmd_args.batch_effect]).size
        ))
    corrected = model._fetch(model.prob_module.softmax_mu, cb.utils.DataDict(data_dict))
    cb.data.write_hybrid_path(time.time() - start_time, f"{cmd_args.output}//time")
    cb.data.write_hybrid_path(corrected, f"{cmd_args.output}//exprs")


if __name__ == "__main__":
    main(parse_args())
    cb.utils.logger.info("Done!")
