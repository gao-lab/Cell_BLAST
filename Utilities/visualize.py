#!/usr/bin/env python

import sys
import argparse

from sklearn.manifold import TSNE
from umap import UMAP

sys.path.append("..")
import Cell_BLAST.data
import Cell_BLAST.message


def dimension_reduction(hybrid_path, method="tSNE", *args, **kwargs):
    x = Cell_BLAST.data.read_hybrid_path(hybrid_path)
    file_name, h5_path = hybrid_path.split("//")
    cached_hybrid_path = "%s//%s/%s" % (file_name, method, h5_path)
    if Cell_BLAST.data.check_hybrid_path(cached_hybrid_path):
        return Cell_BLAST.data.read_hybrid_path(cached_hybrid_path)
    if method == "tSNE":
        result = tsne(x, *args, **kwargs)
    elif method == "UMAP":
        result = umap(x, *args, **kwargs)
    Cell_BLAST.data.write_hybrid_path(result, cached_hybrid_path)
    return result


def tsne(x, perplexity=30, seed=None, *args, **kwargs):
    return TSNE(
        perplexity=perplexity,
        verbose=2, random_state=seed
    ).fit_transform(x)


def umap(x, n_neighbors=15, min_dist=0.1, seed=None, *args, **kwargs):
    return UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        verbose=True, random_state=seed
    ).fit_transform(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-m", "--method", dest="method", type=str, choices=["tSNE", "UMAP"], default="tSNE")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    return parser.parse_args()


def main():
    cmd_args = parse_args()
    dimension_reduction(cmd_args.input, method=cmd_args.method,
                        seed=cmd_args.seed)


if __name__ == "__main__":
    main()
    Cell_BLAST.message.info("Done!")
