#! /usr/bin/env python
# by caozj
# Mar 6, 2020
# 11:37:42 PM

import os
import collections
import pickle
import time
import argparse

import numpy as np
import torch
import scvi.dataset
import scvi.models
import scvi.inference
import scvi.inference.annotation
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, required=True)
    parser.add_argument("-q", "--query", dest="query", type=str, required=True)
    parser.add_argument("-c", "--cutoff", dest="cutoff", type=float, nargs="+", default=[0.95])
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-n", "--normalize", dest="normalize", default=False, action="store_true")
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    cmd_args.output_path = os.path.dirname(cmd_args.output)
    os.makedirs(cmd_args.output_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device or utils.pick_gpu_lowest_memory()
    return cmd_args


@torch.no_grad()
def get_scanvi_class_posterior(scanvi_trainer):
    annotation_posterior = scanvi_trainer.create_posterior()
    scanvi_trainer.model.eval()
    class_posterior = []
    for sample_batch, _, _, _, _ in annotation_posterior:
        class_posterior.append(scanvi_trainer.model.classify(sample_batch))
    return torch.cat(class_posterior).cpu().numpy()


def main(cmd_args):
    print("Loading model...")
    with open(os.path.join(cmd_args.model, "label_encoder.pickle"), "rb") as f:
        label_encoder = pickle.load(f)
    genes = np.loadtxt(os.path.join(cmd_args.model, "genes.txt"), dtype=str)
    vae = torch.load(os.path.join(cmd_args.model, "model.pickle"))

    print("Loading query...")
    query = cb.data.ExprDataSet.read_dataset(cmd_args.query, sparsify=True)
    if cmd_args.clean is not None:
        query = utils.clean_dataset(query, cmd_args.clean)
    n_cells = query.shape[0]
    if cmd_args.normalize:
        query = query.normalize()
    query = query[:, genes]
    query.to_anndata().write_h5ad(os.path.join(cmd_args.output_path, "query.h5ad"))
    query = scvi.dataset.AnnDataset("query.h5ad", save_path=cmd_args.output_path + "/")

    print("Predicting...")
    start_time = time.time()
    trainer = scvi.inference.annotation.CustomSemiSupervisedTrainer(
        vae, query, np.array([]), use_cuda=True, metrics_to_monitor=["ll"])
    prob = get_scanvi_class_posterior(trainer)

    time_per_cell = None
    prediction_dict = collections.defaultdict(lambda: np.repeat("rejected", n_cells).astype(object))
    for cutoff in cmd_args.cutoff:
        mask = prob.max(axis=1) > cutoff
        prediction_dict[cutoff][mask] = label_encoder.inverse_transform(prob[mask].argmax(axis=1))
        if time_per_cell is None:
            time_per_cell = (
                time.time() - start_time
            ) * 1000 / n_cells
    print("Time per cell: %.3fms" % time_per_cell)

    print("Saving result...")
    if os.path.exists(cmd_args.output):
        os.remove(cmd_args.output)
    for cutoff, prediction in prediction_dict.items():
        cb.data.write_hybrid_path(
            prediction, "%s//prediction/%s" % (
                cmd_args.output, str(cutoff)
            )
        )
    cb.data.write_hybrid_path(time_per_cell, "//".join((
        cmd_args.output, "time"
    )))


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
