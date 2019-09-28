#!/usr/bin/env python

import argparse
import sklearn.manifold
import umap
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("--input-slot", dest="input_slot", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("--output-slot", dest="output_slot", type=str, required=True)
    parser.add_argument("-v", "--vis", dest="vis", type=str, choices=["tSNE", "UMAP"], default="tSNE")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    cmd_args = parser.parse_args()
    cmd_args.input = [cmd_args.input]
    cmd_args.output = [cmd_args.output]
    cmd_args.params = argparse.Namespace(
        input_slot=cmd_args.input_slot,
        output_slot=cmd_args.output_slot
    )
    cmd_args.wildcards = argparse.Namespace(
        vis=cmd_args.vis,
        seed=cmd_args.seed
    )
    del cmd_args.input_slot, cmd_args.output_slot, cmd_args.vis, cmd_args.seed
    return cmd_args


def dimension_reduction(x, method="tSNE", **kwargs):
    if method == "tSNE":
        return do_tsne(x, **kwargs)
    if method == "UMAP":
        return do_umap(x, **kwargs)
    raise ValueError("Unknown method!")


def do_tsne(x, perplexity=30, seed=None):
    return sklearn.manifold.TSNE(
        perplexity=perplexity,
        verbose=2, random_state=seed
    ).fit_transform(x)


def do_umap(x, n_neighbors=15, min_dist=0.1, seed=None):
    return umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        verbose=True, random_state=seed
    ).fit_transform(x)


def main():
    x = cb.data.read_hybrid_path("//".join([snakemake.input[0], "latent"]))
    result = dimension_reduction(
        x, method=snakemake.wildcards.vis,
        seed=None if snakemake.wildcards.seed is None else int(snakemake.wildcards.seed)
    )
    cb.data.write_hybrid_path(result, "//".join([snakemake.output[0], "visualization"]))


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
