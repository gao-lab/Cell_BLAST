#! /usr/bin/env python
# by caozj
# Dec 14, 2018
# 2:44:54 PM


import os
import sys
import time
import argparse
import numpy as np
import scipy.stats
import sklearn.preprocessing
import torch
import matplotlib
matplotlib.use("agg")

import scvi.dataset
import scvi.models
import scvi.inference
import scvi.inference.annotation
import Cell_BLAST as cb

sys.path.insert(0, "..")
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
    parser.add_argument("-b", "--batch-effect", dest="batch_effect", type=str, default=None)
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, required=True)

    parser.add_argument("--n-latent", dest="n_latent", type=int, default=10)
    parser.add_argument("--n-hidden", dest="n_hidden", type=int, default=128)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=1)

    parser.add_argument("--supervision", dest="supervision", type=str, default=None)
    parser.add_argument("--label-fraction", dest="label_fraction", type=float, default=None)
    parser.add_argument("--label-priority", dest="label_priority", type=str, default=None)

    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=1000)
    parser.add_argument("--patience", dest="patience", type=int, default=30)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=1e-3)

    parser.add_argument("--no-normalize", dest="no_normalize", default=False, action="store_true")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)
    return cmd_args


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.input, sparsify=True)
    if not cmd_args.no_normalize:
        dataset = dataset.normalize()
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    if cmd_args.genes is not None:
        dataset = dataset[:, dataset.uns[cmd_args.genes]]
    if cmd_args.batch_effect is not None:
        batch_indices = sklearn.preprocessing.LabelEncoder().fit_transform(
            dataset.obs[cmd_args.batch_effect])
    if cmd_args.supervision is not None:
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(
            dataset.obs[cmd_args.supervision])
        if cmd_args.label_fraction is not None:
            if cmd_args.label_priority is not None:
                label_priority = dataset.obs[cmd_args.label_priority]
            else:
                _label_priority = np.random.uniform(size=labels.size)
                label_priority = np.empty(len(_label_priority))
                for l in np.unique(labels):  # Group percentile
                    mask = labels == l
                    label_priority[mask] = (
                        scipy.stats.rankdata(_label_priority[mask]) - 1
                    ) / (mask.sum() - 1)
            if cmd_args.label_fraction == 1.0:
                # Remove a small number of labelled cells to avoid empty
                # unlabelled set, which will lead to a crash.
                cmd_args.label_fraction = 0.99
            labelled_indices = np.where(label_priority >= np.percentile(
                label_priority, (1 - cmd_args.label_fraction) * 100
            ))[0]
        else:
            labelled_indices = np.arange(labels.size)
    dataset.to_anndata().write_h5ad(
        os.path.join(cmd_args.output_path, "data.h5ad"))
    dataset = scvi.dataset.AnnDataset("data.h5ad", save_path=cmd_args.output_path)

    start_time = time.time()
    model_kwargs = dict(
        n_latent=cmd_args.n_latent,
        n_hidden=cmd_args.n_hidden,
        n_layers=cmd_args.n_layers
    )
    trainer_kwargs = dict(
        use_cuda=True, metrics_to_monitor=["ll"], frequency=5,
        early_stopping_kwargs=dict(
            early_stopping_metric="ll", save_best_state_metric="ll",
            patience=cmd_args.patience, threshold=0
        )
    )
    if cmd_args.batch_effect is not None:
        dataset.batch_indices, dataset.n_batches = \
            batch_indices.reshape((-1, 1)), np.unique(batch_indices).size
        model_kwargs["n_batch"] = dataset.n_batches
    if cmd_args.supervision is not None:
        print("Using SCANVI...")
        dataset.labels, dataset.n_labels = \
            labels.reshape((-1, 1)), np.unique(labels).size
        vae = scvi.models.SCANVI(
            dataset.nb_genes, n_labels=dataset.n_labels, **model_kwargs)
        # trainer_kwargs["early_stopping_kwargs"]["on"] = "unlabelled_set"
        trainer = scvi.inference.annotation.CustomSemiSupervisedTrainer(
            vae, dataset, labelled_indices, **trainer_kwargs)
    else:
        print("Using VAE...")
        vae = scvi.models.VAE(dataset.nb_genes, **model_kwargs)
        trainer = scvi.inference.UnsupervisedTrainer(
            vae, dataset, **trainer_kwargs)
    trainer.train(n_epochs=cmd_args.n_epochs, lr=cmd_args.lr)
    cb.data.write_hybrid_path(
        time.time() - start_time,
        os.path.join(cmd_args.output_path, "result.h5//time")
    )
    latent = trainer.get_all_latent_and_imputed_values()["latent"]
    cb.data.write_hybrid_path(
        latent, os.path.join(cmd_args.output_path, "result.h5//latent"))


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
