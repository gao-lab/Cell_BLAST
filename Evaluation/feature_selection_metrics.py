#!/usr/bin/env python

import argparse
import json
import numpy as np
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, required=True)
    parser.add_argument("-r", "--result", dest="result", type=str, required=True)
    parser.add_argument("-l", "--label", dest="label", type=str, required=True)
    parser.add_argument("-n", "--nn", dest="label", type=float, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.output = [cmd_args.output]
    cmd_args.input = argparse.Namespace(
        data=cmd_args.data,
        genes=cmd_args.genes,
        result=cmd_args.result
    )
    cmd_args.config = dict(label=cmd_args.label, nn=cmd_args.nn)
    del cmd_args.data, cmd_args.genes, cmd_args.result, cmd_args.label, cmd_args.nn
    return cmd_args


def main():
    y = cb.data.read_hybrid_path("//".join([
        snakemake.input.data,
        "obs/%s" % snakemake.config["label"]
    ]))
    y = y[~utils.na_mask(y)]
    y = cb.utils.encode_integer(y)[0]

    x = cb.data.read_hybrid_path("//".join([snakemake.input.result, "latent"]))
    performance = dict(
        n_gene=np.loadtxt(snakemake.input.genes, dtype=str).size,
        nearest_neighbor_accuracy=cb.metrics.nearest_neighbor_accuracy(x, y),
        mean_average_precision=cb.metrics.mean_average_precision_from_latent(
            x, y, k=snakemake.config["nn"]),
        time=cb.data.read_hybrid_path(
            "//".join([snakemake.input.result, "time"])),
        n_cell=x.shape[0]
    )

    with open(snakemake.output[0], "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
