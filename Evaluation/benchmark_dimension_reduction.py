#!/usr/bin/env python

"""
This script summarizes model performance of dimension reduction
"""

import sys
import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import tqdm
import Cell_BLAST as cb


def job(pwd):
    __, _, method, dataset, genes, conf, trial = pwd.split("/")
    trial = trial.split("_")[-1]

    result_file = os.path.join(pwd, "result.h5")
    performance_file = os.path.join(pwd, "performance_log.json")
    if not os.path.exists(result_file):
        print("Failed at %s!" % pwd)
        return None  # Some trials may failed, just ignore

    y = cb.data.read_hybrid_path(os.path.join(
        "../Datasets/data", dataset, "data.h5//obs/cell_ontology_class"))
    y = y[~np.in1d(y.astype(str), ("", "na", "NA", "nan", "NaN"))]
    y_checksum = hashlib.md5(y).hexdigest()
    y = cb.utils.encode_integer(y)[0]

    try:
        x = cb.data.read_hybrid_path("%s//latent" % result_file)
        time = cb.data.read_hybrid_path("%s//time" % result_file)
        x_checksum = hashlib.md5(x).hexdigest()
        n_cell = x.shape[0]
    except Exception:
        print(x)
        print(time)
        print("Failed at %s!" % pwd)
        return None

    checksum = x_checksum + y_checksum
    if os.path.exists(performance_file):
        with open(performance_file, "r") as f:
            performance = json.load(f)
    else:
        performance = None
    if performance is None or performance["checksum"] != checksum:
        performance = {"checksum": checksum}

    if "Nearest Neighbor Accuracy" in performance:
        nearest_neighbor_accuracy = performance["Nearest Neighbor Accuracy"]
    else:
        nearest_neighbor_accuracy = cb.metrics.nearest_neighbor_accuracy(x, y)
        performance["Nearest Neighbor Accuracy"] = nearest_neighbor_accuracy

    if "Mean Average Precision" in performance:
        mean_average_precision = performance["Mean Average Precision"]
    else:
        mean_average_precision = cb.metrics.mean_average_precision_from_latent(x, y)
        performance["Mean Average Precision"] = mean_average_precision

    with open(performance_file, "w") as f:
        json.dump(performance, f, indent=4)

    return method, dataset, n_cell, genes, conf, trial, time, \
        nearest_neighbor_accuracy, mean_average_precision


def main():
    methods = os.environ["methods"].split()
    datasets = os.environ["datasets"].split()
    genes = os.environ["genes"]
    dim = "dim_" + os.environ["dims"]

    pwds = []
    for method in methods:
        for dataset in datasets:
            pwd = os.path.join("../Results", method, dataset, genes, dim)
            if os.path.exists(pwd):
                for trial in os.listdir(pwd):
                    if trial.startswith("trial"):
                        pwds.append(os.path.join(pwd, trial))

    rs = joblib.Parallel(
        n_jobs=16, backend="loky",
    )(joblib.delayed(job)(pwd) for pwd in tqdm.tqdm(pwds))
    rs = list(zip(*(item for item in rs if item is not None)))

    df = pd.DataFrame({
        "Method": rs[0],
        "Dataset": rs[1],
        "Number of cells": rs[2],
        "Genes": rs[3],
        "Conf": rs[4],
        "Trial": rs[5],
        "Run time": rs[6],
        "Nearest Neighbor Accuracy": rs[7],
        "Mean Average Precision": rs[8]
    })
    df.to_csv("../Results/benchmark_dimension_reduction.csv", index=False)


if __name__ == "__main__":
    main()
