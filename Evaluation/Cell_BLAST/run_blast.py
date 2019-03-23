#!/usr/bin/env python

import os
import sys
import string
import random
import time
import argparse
import numpy as np
import sklearn.neighbors
import joblib

import Cell_BLAST as cb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, "..")
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", dest="index", type=str, required=True)
    parser.add_argument("-q", "--query", dest="query", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-a", "--annotation", dest="annotation", type=str, default="cell_ontology_class")
    parser.add_argument("-n", "--n-neighbors", dest="n_neighbors", type=int, default=5)
    parser.add_argument("-m", "--min-hits", dest="min_hits", type=int, default=2)
    parser.add_argument("-f", "--filter-by", dest="filter_by", type=str, choices=["dist", "pval"], default="pval")
    parser.add_argument("-c", "--cutoff", dest="cutoff", type=float, nargs="+", default=[0.05])
    parser.add_argument("-l", "--align", dest="align", default=False, action="store_true")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-j", "--n-jobs", dest="n_jobs", type=int, default=1)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("--subsample-ref", dest="subsample_ref", type=int, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    cb.config.RANDOM_SEED = cmd_args.seed
    cb.config.N_JOBS = cmd_args.n_jobs
    return cmd_args


def _fit_nearest_neighbors(x):
    return sklearn.neighbors.NearestNeighbors().fit(x)


def main(cmd_args):

    cb.message.info("Loading index...")
    blast = cb.blast.BLAST.load(
        cmd_args.index, skip_exprs=not cmd_args.align,
        mode="normal" if cmd_args.align else "minimal"
    )
    if cmd_args.subsample_ref is not None:
        cb.message.info("Subsampling reference...")
        subsample_idx = np.random.RandomState(cmd_args.seed).choice(
            blast.ref.shape[0], cmd_args.subsample_ref, replace=False)
        blast.ref = blast.ref[subsample_idx, :]
        blast.clean_latent = blast.clean_latent[:, subsample_idx, ...]
        blast.noisy_latent = blast.noisy_latent[:, subsample_idx, ...]
        blast.nearest_neighbors = joblib.Parallel(n_jobs=cmd_args.n_jobs, backend="loky")(
            joblib.delayed(_fit_nearest_neighbors)(
                _clean_latent
            ) for _clean_latent in blast.clean_latent
        )

    cb.message.info("Reading query...")
    query = cb.data.ExprDataSet.read_dataset(cmd_args.query).normalize()
    if cmd_args.clean:
        query = utils.clean_dataset(query, cmd_args.clean)

    if cmd_args.align:
        cb.message.info("Aligning...")
        unipath = "/tmp/cb/" + "".join(random.choices(
            string.ascii_uppercase + string.digits, k=32))
        cb.message.info("Using temporary path: " + unipath)
        blast = blast.align(query, path=unipath).build_empirical()

    cb.message.info("BLASTing...")
    start_time = time.time()
    hits = blast.query(
        query, n_neighbors=cmd_args.n_neighbors
    ).reconcile_models()

    time_per_cell = None
    prediction_dict = {}
    for cutoff in cmd_args.cutoff:
        prediction_dict[cutoff] = hits.filter(
            by=cmd_args.filter_by, cutoff=cutoff
        ).annotate(
            cmd_args.annotation, min_hits=cmd_args.min_hits
        )[cmd_args.annotation]
        if time_per_cell is None:
            time_per_cell = (
                time.time() - start_time
            ) * 1000 / len(prediction_dict[cutoff])
    print("Time per cell: %.3fms" % time_per_cell)

    cb.message.info("Saving result...")
    if os.path.exists(cmd_args.output):
        os.remove(cmd_args.output)
    for cutoff in prediction_dict:
        cb.data.write_hybrid_path(
            prediction_dict[cutoff], "%s//prediction/%s" % (
                cmd_args.output, str(cutoff)
            )
        )
    cb.data.write_hybrid_path(time_per_cell, "//".join((
        cmd_args.output, "time"
    )))


if __name__ == "__main__":
    main(parse_args())
