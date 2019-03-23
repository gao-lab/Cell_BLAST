#! /usr/bin/env python
# by caozj
# Nov 1, 2018
# 9:16:57 PM

import os
import sys
import argparse
import json

import numpy as np
import h5py

sys.path.append("../..")
import Cell_BLAST.message
import Cell_BLAST.utils
import Cell_BLAST.data
import run_classifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, required=True)
    parser.add_argument("-q", "--query", dest="query", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-d", "--device", dest="device", type=str, choices=["", "0", "1", "2", "3"], default="")
    cmd_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if not os.path.exists(os.path.dirname(cmd_args.output)):
        os.makedirs(os.path.dirname(cmd_args.output))
    return cmd_args


def main():
    cmd_args = parse_args()

    Cell_BLAST.message.info("Building model...")
    with open(os.path.join(cmd_args.model, "cmd_args.json"), "r") as f:
        model_args = Cell_BLAST.utils.dotdict(json.load(f))
    model = run_classifier.build_model(model_args)

    Cell_BLAST.message.info("Reading query...")
    query = Cell_BLAST.data.ExprDataSet.read_dataset(cmd_args.query)
    query = query.normalize()
    query.exprs = np.log1p(query.exprs)
    query = query[:, Cell_BLAST.data.read_hybrid_path("%s//uns/%s" % (
        model_args.input, model_args.genes
    ))]
    query_latent = model.inference(query.exprs)

    query_pred = model.classify(query.exprs)
    query_class = query_pred.argmax(axis=1)
    query_confidence = query_pred.max(axis=1)
    query_confidence = -np.log(1 + 1e-8 - query_confidence)

    with h5py.File(cmd_args.output, "w") as f:
        f.create_dataset("latent", data=query_latent)
        f.create_dataset("class", data=query_class)
        f.create_dataset("confidence", data=query_confidence)


if __name__ == "__main__":
    main()
    Cell_BLAST.message.info("Done!")
