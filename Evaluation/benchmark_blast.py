#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import h5py
import sklearn.metrics
import tqdm
import Cell_BLAST as cb
import utils


def get_jobs():
    methods = os.environ["methods"].split()
    datasets = os.environ["datasets"].split()
    dim = os.environ["dims"]
    lambda_rmbatch_reg = os.environ["lambda_rmbatch_regs"]

    _datasets = {}
    for ref, query in (item.split(":") for item in datasets):
        if ref not in _datasets:
            _datasets[ref] = [query]
        else:
            _datasets[ref].append(query)

    for method in methods:
        for dataset in _datasets:  # reference
            if method == "scmap":
                pwd = os.path.join("../Results", method, dataset, "scmap_genes")
            elif method == "CellFishing.jl":
                pwd = os.path.join("../Results", method, dataset, "cf_genes")
            else:
                conf = "dim_%s" % dim
                if dataset.find("+") >= 0:
                    conf += "_rmbatch_" + lambda_rmbatch_reg
                pwd = os.path.join("../Results", method, dataset, "seurat_genes", conf)
            for trial in os.listdir(pwd):
                if method in ("scmap", "CellFishing.jl"):
                    if trial.startswith("trial_"):
                        yield os.path.join(pwd, trial), _datasets[dataset]
                elif method == "Cell_BLAST":
                    if trial.startswith("blast_") and not trial.endswith(".bak"):
                        yield os.path.join(pwd, trial), _datasets[dataset]


def do_job(pwd, queries, aligned=False):
    pwd_split = pwd.split("/")
    if len(pwd_split) == 6:  # scmap and CellFishing.jl
        if aligned:
            return None
        __, _, method, dataset, genes, trial = pwd_split
    else:  # Cell_BLAST
        __, _, method, dataset, genes, conf, trial = pwd_split

    # Read reference
    ref_cl = cb.data.read_hybrid_path(os.path.join(
        "../Datasets/data", dataset, "data.h5//obs/cell_ontology_class"))
    ref_organ = cb.data.read_hybrid_path(os.path.join(
        "../Datasets/data", dataset, "data.h5//obs/organ"))
    ref_mask = utils.na_mask(ref_cl)
    ref_cl, ref_organ = ref_cl[~ref_mask], ref_organ[~ref_mask]
    ref_size, pos_cl, pos_organ = len(ref_cl), np.unique(ref_cl), np.unique(ref_organ)

    # Read query
    true, pred_dict, organ, time = [], {}, [], []
    suffix = "_aligned" if aligned else ""
    for query in queries:
        try:
            _true = cb.data.read_hybrid_path(os.path.join(
                "../Datasets/data", query, "data.h5//obs/cell_ontology_class"))
            _organ = cb.data.read_hybrid_path(os.path.join(
                "../Datasets/data", query, "data.h5//obs/organ"))
            _mask = utils.na_mask(_true)
            _true, _organ = _true[~_mask], _organ[~_mask]
            with h5py.File(os.path.join(
                pwd, "%s.h5" % (query + suffix)
            ), "r") as f:
                fpred = f["prediction"]
                for threshold in fpred:
                    _pred = fpred[threshold][...].ravel()
                    if _pred.dtype.type is np.bytes_:
                        _pred = cb.utils.decode(_pred)
                    if threshold not in pred_dict:
                        pred_dict[threshold] = [_pred]
                    else:
                        pred_dict[threshold].append(_pred)
            _time = cb.data.read_hybrid_path(os.path.join(
                pwd, "%s.h5//time" % (query + suffix)))
            true.append(_true)
            organ.append(_organ)
            time.append(_time)
        except Exception:
            print("Failed at %s with %s!" % (pwd, query + suffix))
    if len(time) == 0:
        return None
    weight = np.concatenate([np.repeat(1 / item.size, item.size) for item in true])
    weight /= weight.sum() / weight.size
    true = np.concatenate(true)
    organ = np.concatenate(organ)
    pred_dict = {k: np.concatenate(v) for k, v in pred_dict.items()}
    # weight = {item: 1 / count for item, count in zip(*np.unique(true, return_counts=True))}
    # weight = np.array([weight[item] for item in true])
    # weight /= weight.sum() / weight.size

    tp_cl = np.in1d(true, pos_cl)
    tp_organ = np.in1d(organ, np.unique(pos_organ))
    tn_cl, tn_organ = ~tp_cl, ~tp_organ

    sensitivity, specificity, kappa, extended_kappa, threshold = [], [], [], [], []
    for thresh, pred in pred_dict.items():
        threshold.append(float(thresh))
        pn = np.vectorize(
            lambda x: x in ("unassigned", "ambiguous", "rejected")
        )(pred)
        pp = ~pn
        sensitivity.append((weight * np.logical_and(tp_cl, pp)).sum() / (weight * tp_cl).sum())
        specificity.append((weight * np.logical_and(tn_cl, pn)).sum() / (weight * tn_cl).sum())
        if sensitivity[-1] > 0:
            _kappa = sklearn.metrics.cohen_kappa_score(
                true[np.logical_and(tp_cl, pp)],
                pred[np.logical_and(tp_cl, pp)],
                sample_weight=weight[np.logical_and(tp_cl, pp)]
            )
            if np.isnan(_kappa):  # Only one category, and correctly predicted
                _kappa = 1.0
        else:
            _kappa = 0.0
        kappa.append(_kappa)
        pred_cp, true_cp = pred.copy(), true.copy()
        pred_cp[pn], true_cp[tn_cl] = "REJECT", "REJECT"
        extended_kappa.append(sklearn.metrics.cohen_kappa_score(true_cp, pred_cp))
    assert len(sensitivity) == len(specificity) == len(kappa) == len(extended_kappa) == len(threshold) == len(pred_dict)
    assert np.unique(ref_organ).size == 1
    ref_organ = ref_organ[0]

    return [method + suffix] * len(pred_dict), [dataset] * len(pred_dict), [ref_organ] * len(pred_dict), \
        [ref_size] * len(pred_dict), [trial] * len(pred_dict), [np.mean(time)] * len(pred_dict), \
        sensitivity, specificity, kappa, extended_kappa, threshold


def main():
    rs = []
    jobs = list(get_jobs())
    for pwd, queries in tqdm.tqdm(jobs):  # Cannot do multiprocessing because of pickling issues
        rs.append(do_job(pwd, queries, False))
        rs.append(do_job(pwd, queries, True))
    rs = [np.concatenate(item) for item in zip(*(item for item in rs if item is not None))]
    df = pd.DataFrame({
        "Method": rs[0],
        "Dataset": rs[1],
        "Organ": rs[2],
        "Reference size": rs[3],
        "Trial": rs[4],
        "Time per query": rs[5],
        "Sensitivity": rs[6],
        "Specificity": rs[7],
        "Kappa": rs[8],
        "Extended Kappa": rs[9],
        "Threshold": rs[10]
    })
    df.to_csv("../Results/benchmark_blast.csv", index=False)


if __name__ == "__main__":
    main()
