#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--n-genes", dest="n_genes", type=int, default=1000)
    parser.add_argument("--mu-shape", dest="mu_shape", type=float, default=0.2)
    parser.add_argument("--mu-scale", dest="mu_scale", type=float, default=50.0)
    parser.add_argument("--theta-shape", dest="theta_shape", type=float, default=10.0)
    parser.add_argument("--theta-scale", dest="theta_scale", type=float, default=0.1)
    parser.add_argument("--p-alpha", dest="p_alpha", type=float, default=0.5)
    parser.add_argument("--p-beta", dest="p_beta", type=float, default=3.0)
    parser.add_argument("-f", "--type-freqs", dest="type_freqs", nargs="+", type=float, default=[1.0])
    parser.add_argument("-n", "--n-cells", dest="n_cells", type=int, default=10000)
    parser.add_argument("-l", "--latent-dim", dest="latent_dim", type=int, default=10)
    parser.add_argument("-c", "--cat-dim", dest="cat_dim", type=int, default=10)
    parser.add_argument("-v", "--visualization", dest="visualization", type=str, choices=["tSNE", "UMAP"], default="tSNE")
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    np.random.seed(cmd_args.seed)
    cb.config.RANDOM_SEED = cmd_args.seed
    cmd_args.type_freqs = np.array(cmd_args.type_freqs).astype(np.float)
    cmd_args.type_freqs /= cmd_args.type_freqs.sum()
    if cmd_args.latent_dim == 2:
        cmd_args.visualization = None
    return cmd_args


def main(cmd_args):

    # Simulation
    cb.message.info("Simulating expression matrix...")
    exprs_list, type_list = [], []
    for i in range(len(cmd_args.type_freqs)):
        type_size = np.round(cmd_args.type_freqs[i] * cmd_args.n_cells).astype(int)
        nb_mu = np.random.gamma(shape=cmd_args.mu_shape, scale=cmd_args.mu_scale, size=cmd_args.n_genes)
        nb_theta = np.random.gamma(shape=cmd_args.theta_shape, scale=cmd_args.theta_scale, size=cmd_args.n_genes)
        zi_p = np.random.beta(a=cmd_args.p_alpha, b=cmd_args.p_beta, size=cmd_args.n_genes)
        nb_p, nb_n = nb_theta / (nb_mu + nb_theta), nb_theta
        zinb_samples = np.stack([
            np.random.negative_binomial(n=nb_n, p=nb_p)
            for _ in range(type_size)
        ], axis=0)
        zi_mask = np.stack([
            np.random.binomial(n=1, p=zi_p)
            for _ in range(type_size)
        ], axis=0).astype(bool)
        zinb_samples[zi_mask] = 0.0
        exprs_list.append(zinb_samples)
        type_list.append(np.repeat("type_%d" % i, type_size))

    dataset = cb.data.ExprDataSet(
        exprs=np.concatenate(exprs_list, axis=0),
        obs=pd.DataFrame(dict(
            type=np.concatenate(type_list, axis=0)
        ), index=np.arange(cmd_args.n_cells)),
        var=pd.DataFrame(index=np.arange(cmd_args.n_genes)),
        uns={}
    )

    # Model fitting
    cb.message.info("Fitting model...")
    data_dict = cb.utils.DataDict()
    data_dict["x"] = dataset.exprs
    model = cb.directi.fit_DIRECTi(
        dataset, latent_dim=cmd_args.latent_dim, cat_dim=cmd_args.cat_dim,
        epoch=50, patience=50, path=cmd_args.output_path
    )
    model.save()

    # Model evaluation
    cb.message.info("Evaluating model...")
    dataset.latent = model.inference(dataset)
    dataset.obs.insert(0, "cluster", np.vectorize(
        lambda x: "cluster_%d" % x
    )(model.clustering(dataset)[0]))
    plot = dataset.visualize_latent(
        "type", method=cmd_args.visualization, width=5.5, height=5, sort=True)
    plot.get_figure().savefig(os.path.join(
        cmd_args.output_path, "type_visualization.pdf"
    ), bbox_inches="tight")
    plot = dataset.visualize_latent(
        "cluster", method=cmd_args.visualization, width=5.5, height=5, sort=True)
    plot.get_figure().savefig(os.path.join(
        cmd_args.output_path, "cluster_visualization.pdf"
    ), bbox_inches="tight")
    dataset.write_dataset(os.path.join(cmd_args.output_path, "data.h5"))


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
