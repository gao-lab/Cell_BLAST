#!/usr/bin/env python

import os
import argparse
import collections
import json
import numpy as np
import pandas as pd
import h5py
# import sklearn.metrics
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", dest="ref", type=str, nargs="+")
    parser.add_argument("-t", "--true", dest="true", type=str, nargs="+")
    parser.add_argument("-p", "--pred", dest="pred", type=str, nargs="+")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-c", "--cell-type-specific", dest="cell_type_specific", type=str, required=True)
    parser.add_argument("-l", "--label", dest="label", type=str, required=True)
    parser.add_argument("-e", "--expect", dest="expect", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.output = [cmd_args.output, cmd_args.cell_type_specific]
    cmd_args.input = argparse.Namespace(
        ref=cmd_args.ref,
        true=cmd_args.true,
        pred=cmd_args.pred
    )
    cmd_args.config = argparse.Namespace(label=cmd_args.label)
    cmd_args.params = argparse.Namespace(expect=cmd_args.expect)
    del cmd_args.red, cmd_args.true, cmd_args.pred, cmd_args.label, \
        cmd_args.expect, cmd_args.cell_type_specific
    return cmd_args


def main():
    ref = np.concatenate([cb.data.read_hybrid_path("{file}//obs/{label}".format(
        file=item, label=snakemake.config["label"]
    )) for item in snakemake.input.ref])
    ref = ref[~utils.na_mask(ref)]
    pos_types = np.unique(ref)

    expect = pd.read_csv(snakemake.params.expect, index_col=0)

    # # Pos/neg weighed
    # true = np.concatenate([cb.data.read_hybrid_path("{file}//obs/{label}".format(
    #     file=item, label=snakemake.config["label"]
    # )) for item in snakemake.input.true])
    # true = true[~utils.na_mask(true)]
    # tp = np.in1d(true, pos_types)
    # tn = ~tp
    # weight = np.ones(true.size)
    # weight[tp] = 1 / tp.sum()
    # weight[tn] = 1 / tn.sum()
    # weight /= weight.sum() / weight.size

    # Dataset weighed
    true = [cb.data.read_hybrid_path("{file}//obs/{label}".format(
        file=item, label=snakemake.config["label"]
    )) for item in snakemake.input.true]
    true = [item[~utils.na_mask(item)] for item in true]
    weight = np.concatenate([np.repeat(1 / item.size, item.size) for item in true])
    weight /= weight.sum() / weight.size
    true = np.concatenate(true)
    tp = np.in1d(true, pos_types)
    tn = ~tp

    pred_dict = collections.defaultdict(list)
    for item in snakemake.input.pred:
        with h5py.File(item, "r") as f:
            g = f["prediction"]
            for threshold in g:
                pred_dict[float(threshold)].append(
                    cb.data.read_clean(g[threshold][...]))

    cell_type_specific_excel = pd.ExcelWriter(snakemake.output[1])
    performance = []
    for threshold in sorted(pred_dict.keys(), key=float):
        pred = pred_dict[threshold] = np.concatenate(pred_dict[threshold])
        assert len(pred) == len(true)
        pn = np.vectorize(
            lambda x: x in ("unassigned", "ambiguous", "rejected")
        )(pred)
        pp = ~pn
        sensitivity = (weight * np.logical_and(tp, pp)).sum() / (weight * tp).sum()
        specificity = (weight * np.logical_and(tn, pn)).sum() / (weight * tn).sum()
        class_specific_accuracy = cb.metrics.class_specific_accuracy(true, pred, expect)
        class_specific_accuracy.insert(0, "positive", np.in1d(class_specific_accuracy.index, pos_types))
        pos_mba = class_specific_accuracy.loc[class_specific_accuracy["positive"], "accuracy"].mean()
        neg_mba = class_specific_accuracy.loc[~class_specific_accuracy["positive"], "accuracy"].mean()
        mba = (pos_mba + neg_mba) / 2
        performance.append(dict(
            ref_size=ref.size,
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            pos_mba=pos_mba,
            neg_mba=neg_mba,
            mba=mba
        ))
        class_specific_accuracy.to_excel(
            cell_type_specific_excel, str(threshold),
            index_label=snakemake.config["label"]
        )
        cell_type_specific_excel.save()

    with open(snakemake.output[0], "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
