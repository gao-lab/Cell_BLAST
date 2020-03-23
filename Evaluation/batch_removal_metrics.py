#!/usr/bin/env python

import argparse
import json
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, required=True)
    parser.add_argument("-r", "--result", dest="result", type=str, required=True)
    parser.add_argument("-l", "--label", dest="label", type=str, required=True)
    parser.add_argument("-b", "--batch", dest="batch", type=str, required=True)
    parser.add_argument("-n", "--nn", dest="nn", type=float, required=True)
    parser.add_argument("-s", "--slot", dest="slot", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.output = [cmd_args.output]
    cmd_args.input = argparse.Namespace(
        data=cmd_args.data,
        result=cmd_args.result
    )
    cmd_args.config = dict(
        label=cmd_args.label,
        batch=cmd_args.batch,
        nn=cmd_args.nn
    )
    cmd_args.params = argparse.Namespace(
        slot=cmd_args.slot
    )
    del cmd_args.data, cmd_args.result, cmd_args.label, cmd_args.batch, cmd_args.nn, cmd_args.slot
    return cmd_args


def main():
    y = cb.data.read_hybrid_path("//".join([
        snakemake.input.data,
        "obs/%s" % snakemake.config["label"]
    ]))
    mask = utils.na_mask(y)
    y = y[~mask]
    y = cb.utils.encode_integer(y)[0]
    b = cb.data.read_hybrid_path("//".join([
        snakemake.input.data,
        "obs/%s" % snakemake.config["batch"]
    ]))
    b = b[~mask]
    b = cb.utils.encode_integer(b)[0]
    x = cb.data.read_hybrid_path("//".join([snakemake.input.result, snakemake.params.slot]))

    performance = dict(
        nearest_neighbor_accuracy=cb.metrics.nearest_neighbor_accuracy(x, y),
        mean_average_precision=cb.metrics.mean_average_precision_from_latent(x, y, k=snakemake.config["nn"]),
        seurat_alignment_score=cb.metrics.seurat_alignment_score(x, b, n=10, k=snakemake.config["nn"]),
        batch_mixing_entropy=cb.metrics.batch_mixing_entropy(x, b),
        time=float(cb.data.read_hybrid_path("//".join([snakemake.input.result, "time"]))),  # "Null" have time = 0 read as np.int64
        n_cell=x.shape[0]
    )

    with open(snakemake.output[0], "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
