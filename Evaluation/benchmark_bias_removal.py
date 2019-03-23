#!/usr/bin/env python

"""
This script summarizes model performance of bias removal
"""

import sys
import os
import re
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import tqdm
import Cell_BLAST as cb


def job(pwd):
    __, _, method, dataset, genes, conf, trial = pwd.split("/")
    rmbatch_reg = conf.split("rmbatch_")[-1] \
        if len(conf.split("rmbatch_")) > 1 else ""
    trial = trial.split("_")[-1]

    result_file = os.path.join(pwd, "result.h5")
    performance_file = os.path.join(pwd, "performance_log.json")
    if not os.path.exists(result_file):
        print("Failed at %s!" % pwd)
        return None  # Some trials may failed, just ignore

    y = cb.data.read_hybrid_path(os.path.join(
        "../Datasets/data", dataset, "data.h5//obs/cell_ontology_class"))
    mask = np.in1d(y.astype(str), ("", "na", "NA", "nan", "NaN"))
    y = y[~mask]
    y_checksum = hashlib.md5(y).hexdigest()
    y = cb.utils.encode_integer(y)[0]

    b = cb.data.read_hybrid_path(os.path.join(
        "../Datasets/data", dataset, "data.h5//obs/%s" % os.environ["batch"]))
    b = b[~mask]
    b_checksum = hashlib.md5(b).hexdigest()
    b = cb.utils.encode_integer(b)[0]

    try:
        x = cb.data.read_hybrid_path("%s//latent" % result_file)
        time = cb.data.read_hybrid_path("%s//time" % result_file)
        x_checksum = hashlib.md5(x).hexdigest()
        n_cell = x.shape[0]
    except Exception:
        print("Failed at %s!" % pwd)
        return None

    checksum = x_checksum + y_checksum + b_checksum
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            performance = json.load(f)
    else:
        performance = None
    if performance is None or performance["checksum"] != checksum:
        performance = {"checksum": checksum}

    if "Mean Average Precision" in performance:
        mean_average_precision = performance["Mean Average Precision"]
    else:
        mean_average_precision = cb.metrics.mean_average_precision_from_latent(x, y)
        performance["Mean Average Precision"] = mean_average_precision

    if "Seurat Alignment Score" in performance:
        seurat_alignment_score = performance["Seurat Alignment Score"]
    else:
        seurat_alignment_score = cb.metrics.seurat_alignment_score(x, b, n=10)
        # FIXME: 10 repeats might not be enough
        performance["Seurat Alignment Score"] = seurat_alignment_score

    if "Batch Mixing Entropy" in performance:
        batch_mixing_entropy = performance["Batch Mixing Entropy"]
    else:
        batch_mixing_entropy = cb.metrics.batch_mixing_entropy(x, b)
        performance["Batch Mixing Entropy"] = batch_mixing_entropy

    with open(performance_file, "w") as f:
        json.dump(performance, f, indent=4)

    return method, dataset, n_cell, genes, rmbatch_reg, trial, time, \
        mean_average_precision, seurat_alignment_score, batch_mixing_entropy


def main():

    pwds = []
    for method in os.environ["methods"].split():
        for dataset in os.environ["datasets"].split():
            pwd = os.path.join("../Results", method, dataset, os.environ["genes"])
            if os.path.exists(pwd):
                for conf in [
                    item for item in os.listdir(pwd) if not item.startswith(".") and ((
                        len(re.findall("rmbatch_", item, flags=re.IGNORECASE)) > 0 and
                        len(re.findall("sup_", item, flags=re.IGNORECASE)) == 0
                    ) or method != "Cell_BLAST")
                ]:  # FIXME: not start with "." because hid directories as backup
                    for trial in os.listdir(os.path.join(pwd, conf)):
                        if trial.startswith("trial_"):
                            pwds.append(os.path.join(pwd, conf, trial))

    rs = joblib.Parallel(
        n_jobs=16, backend="loky"
    )(joblib.delayed(job)(pwd) for pwd in tqdm.tqdm(pwds))
    rs = list(zip(*(item for item in rs if item is not None)))

    df = pd.DataFrame({
        "Method": rs[0], "Dataset": rs[1], "Number of cells": rs[2],
        "Genes": rs[3], "Regularization": rs[4], "Trial": rs[5], "Run time": rs[6],
        "Mean Average Precision": rs[7], "Seurat Alignment Score": rs[8],
        "Batch Mixing Entropy": rs[9]
    })
    df.to_csv("../Results/benchmark_bias_removal.csv", index=False)


if __name__ == "__main__":
    main()
