#! /usr/bin/env python
# by caozj
# 20 Dec 2017
# 10:34:35 AM


import os
import sys
import argparse
import time

import numpy as np
import scipy.stats

import Cell_BLAST as cb
sys.path.insert(0, "..")
import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-i", "--input", dest="input", type=str, default=None)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, default=None)

    # Architecture
    parser.add_argument("-l", "--latent-dim", dest="latent_dim", type=int, default=10)
    parser.add_argument("-c", "--cat-dim", dest="cat_dim", type=int, default=None)
    parser.add_argument("--h-dim", dest="h_dim", type=int, default=128)
    parser.add_argument("--depth", dest="depth", type=int, default=1)
    parser.add_argument("--prob-module", dest="prob_module", choices=["NB", "ZINB", "ZIG"], default="NB")

    # Remove systematical bias
    parser.add_argument("-b", "--batch-effect", dest="batch_effect", type=str, default=None)
    parser.add_argument("--rmbatch-module", dest="rmbatch_module", choices=["RMBatch", "Adversarial", "MNN", "MNNAdversarial"], default="Adversarial")

    # Supervision
    parser.add_argument("--supervision", dest="supervision", type=str, default=None)
    parser.add_argument("--label-fraction", dest="label_fraction", type=float, default=None)
    parser.add_argument("--label-priority", dest="label_priority", type=str, default=None)

    # Regularization strength
    parser.add_argument("--lambda-prior-reg", dest="lambda_prior_reg", type=float, default=0.001)
    parser.add_argument("--lambda-prob-reg", dest="lambda_prob_reg", type=float, default=0.01)
    parser.add_argument("--lambda-rmbatch-reg", dest="lambda_rmbatch_reg", type=float, default=0.01)
    parser.add_argument("--lambda-sup", dest="lambda_sup", type=float, default=10.0)

    # Learning
    parser.add_argument("--val-split", dest="val_split", type=float, default=0.2)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--optimizer", dest="optimizer", type=str, default="RMSPropOptimizer")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--epoch", dest="epoch", type=int, default=1000)
    parser.add_argument("--patience", dest="patience", type=int, default=30)

    # Misc
    parser.add_argument("--no-normalize", dest="no_normalize", default=False, action="store_true")
    parser.add_argument("-n", "--no-fit", dest="no_fit", default=False, action="store_true")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, choices=["", "0", "1", "2", "3"], default="")
    parser.add_argument("--clean", dest="clean", type=str, default=None)

    cmd_args = cb.utils.dotdict(vars(parser.parse_args()))

    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.input is None or cmd_args.output_path is None:
        raise ValueError("`-i` and `-o` must be specified!")
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    with open(os.path.join(cmd_args.output_path, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    return cmd_args


def main(cmd_args):
    result_file = os.path.join(cmd_args.output_path, "result.h5")

    cb.message.info("Reading data...")
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.input)
    if not cmd_args.no_normalize:
        dataset = dataset.normalize()
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)

    if cmd_args.supervision is not None and cmd_args.label_fraction is not None:
        label = dataset.obs[cmd_args.supervision]
        if cmd_args.label_priority is not None:
            label_priority = dataset.obs[cmd_args.label_priority].values
        else:
            _label_priority = np.random.uniform(size=label.shape[0])
            label_priority = np.empty(len(_label_priority))
            for l in np.unique(label):  # Group percentile
                mask = label == l
                label_priority[mask] = (
                    scipy.stats.rankdata(_label_priority[mask]) - 1
                ) / (mask.sum() - 1)
        exclude_mask = label_priority < np.percentile(
            label_priority, (1 - cmd_args.label_fraction) * 100)
        dataset.obs.loc[exclude_mask, cmd_args.supervision] = np.nan

        cb.data.write_hybrid_path(
            ~exclude_mask, "%s//supervision" % result_file)

    latent_module_kwargs = dict(lambda_reg=cmd_args.lambda_prior_reg)
    if cmd_args.supervision is not None:
        latent_module_kwargs["lambda_sup"] = cmd_args.lambda_sup
    prob_module_kwargs = dict(lambda_reg=cmd_args.lambda_prob_reg)
    rmbatch_module_kwargs = dict(lambda_reg=cmd_args.lambda_rmbatch_reg)

    start_time = time.time()
    model = cb.directi.fit_DIRECTi(
        dataset, genes=None if cmd_args.genes is None else dataset.uns[cmd_args.genes],
        latent_dim=cmd_args.latent_dim, cat_dim=cmd_args.cat_dim,
        supervision=cmd_args.supervision, batch_effect=cmd_args.batch_effect,
        h_dim=cmd_args.h_dim, depth=cmd_args.depth,
        prob_module=cmd_args.prob_module, rmbatch_module=cmd_args.rmbatch_module,
        latent_module_kwargs=latent_module_kwargs,
        prob_module_kwargs=prob_module_kwargs,
        rmbatch_module_kwargs=rmbatch_module_kwargs,
        optimizer=cmd_args.optimizer, learning_rate=cmd_args.learning_rate,
        batch_size=cmd_args.batch_size, val_split=cmd_args.val_split,
        epoch=cmd_args.epoch, patience=cmd_args.patience,
        progress_bar=True, random_seed=cmd_args.seed, path=cmd_args.output_path
    )
    model.save()

    cb.message.info("Saving results...")
    dataset = dataset[:, model.genes]
    inferred_latent = model.inference(dataset.exprs)
    cb.data.write_hybrid_path(
        time.time() - start_time, "%s//time" % result_file)
    cb.data.write_hybrid_path(
        inferred_latent, "%s//latent" % result_file)
    try:  # If intrinsic clustering is used
        cb.data.write_hybrid_path(
            model.clustering(dataset.exprs)[0],
            "%s//cluster" % result_file
        )
    except Exception:
        pass


if __name__ == "__main__":
    main(parse_args())
    cb.message.info("Done!")
