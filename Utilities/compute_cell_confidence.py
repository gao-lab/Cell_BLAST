#! /usr/bin/env python
# by caozj
# Oct 22, 2018
# 10:08:23 PM

"""
This script determines which cells in a dataset are more confidently labeled
Methods to try out:
* weakly trained SVM
* average silhouette score based on cosine similarity (pretty good)
* k-means + best match cluster assignment + unmatch cells (not implemented)
"""

import sys
import argparse

sys.path.insert(0, "../..")
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default="seurat_genes")
    parser.add_argument("-l", "--label", dest="label", type=str, default="cell_type1")
    return parser.parse_args()


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.dataset).normalize()
    conf, nconf = dataset.annotation_confidence(
        cmd_args.label, cmd_args.genes, return_group_percentile=True)
    cb.data.write_hybrid_path(
        conf, "%s//obs/confidence" % cmd_args.dataset)
    cb.data.write_hybrid_path(
        nconf, "%s//obs/normalized_confidence" % cmd_args.dataset)


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")