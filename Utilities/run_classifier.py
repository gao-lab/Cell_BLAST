#! /usr/bin/env python
# by caozj
# Oct 30, 2018
# 9:37:33 PM


import sys
import os
import argparse
import time
import random
import json
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as skp
import h5py

sys.path.append("../..")
import Cell_BLAST.utils
import Cell_BLAST.data
import Cell_BLAST.classifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, required=True)

    # Architecture
    parser.add_argument("-l", "--latent-dim", dest="latent_dim", type=int, default=10)
    parser.add_argument("--h-dim", dest="h_dim", type=int, default=512)
    parser.add_argument("--depth", dest="depth", type=int, default=1)

    # Learning
    parser.add_argument("--val-split", dest="val_split", type=float, default=0.1)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--optimizer", dest="optimizer", type=str, default="RMSPropOptimizer")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--epoch", dest="epoch", type=int, default=1000)
    parser.add_argument("--patience", dest="patience", type=int, default=30)

    # Supervision
    parser.add_argument("--supervision", dest="supervision", type=str, required=True)
    parser.add_argument("--label-fraction", dest="label_fraction", type=float, default=1.)
    parser.add_argument("--label-priority", dest="label_priority", type=str, default=None)

    # Misc
    parser.add_argument("-n", "--no-fit", dest="no_fit", default=False, action="store_true")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, choices=["", "0", "1", "2", "3"], default="")

    cmd_args = parser.parse_args()
    cmd_args = Cell_BLAST.utils.dotdict(vars(cmd_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device

    if cmd_args.seed is not None:
        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)

    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    return cmd_args


def read_data(conf):
    data_dict = Cell_BLAST.utils.DataDict()
    dataset = Cell_BLAST.data.ExprDataSet.read_dataset(conf.input)

    dataset = dataset.normalize()
    if conf.genes is not None:
        dataset = dataset[:, dataset.uns[conf.genes]]
    label = skp.OneHotEncoder().fit_transform(
        skp.LabelEncoder().fit_transform(
            dataset.obs[conf.supervision]
        ).reshape((-1, 1))
    )
    data_dict["x"] = np.log1p(dataset.exprs)
    data_dict["y"] = label

    conf.x_dim = data_dict["x"].shape[1]
    conf.n_class = data_dict["y"].shape[1]

    if conf.label_priority:
        include_idx = np.where(
            dataset.obs[conf.label_priority].values >=
            np.percentile(
                dataset.obs[conf.label_priority].values,
                (1 - conf.label_fraction) * 100
            )
        )[0]
    else:
        include_idx = np.empty(0, dtype=np.int64)
        for i in range(label.shape[1]):  # Stratified label removal
            include_idx = np.concatenate([np.random.choice(
                np.where(label[:, i].toarray())[0],
                np.ceil(label[:, i].sum() * (
                    conf.label_fraction
                )).astype(int), replace=False
            ), include_idx])

    return data_dict, data_dict[include_idx]


def build_model(conf):
    model = Cell_BLAST.classifier.Classifier(
        path=conf.output_path,
        x_dim=conf.x_dim, latent_dim=conf.latent_dim, n_class=conf.n_class,
        h_dim=conf.h_dim, depth=conf.depth
    )
    model.compile(optimizer=tf.train.__dict__[conf.optimizer],
                  lr=conf.learning_rate)
    if os.path.exists(os.path.join(model.path, "final")):
        Cell_BLAST.message.info("Loading existing weights...")
        model.load(os.path.join(model.path, "final"))
    return model


def main(cmd_args):
    result_file = os.path.join(cmd_args.output_path, "result.h5")

    Cell_BLAST.message.info("Reading data...")
    full_data_dict, sup_data_dict = read_data(cmd_args)

    with open(os.path.join(cmd_args.output_path, "cmd_args.json"), "w") as f:
        cmd_args.cmd = " ".join(sys.argv)
        json.dump(cmd_args, f, indent=4)

    Cell_BLAST.message.info("Building model...")
    model = build_model(cmd_args)

    if not cmd_args.no_fit:
        Cell_BLAST.message.info("Fitting model...")
        start_time = time.time()
        model.fit(sup_data_dict, val_split=cmd_args.val_split,
                  epoch=cmd_args.epoch, patience=cmd_args.patience,
                  batch_size=cmd_args.batch_size)
        model.save(os.path.join(model.path, "final"))
        with h5py.File(result_file, "a") as f:
            if "tSNE" in f:
                del f["tSNE"]
            if "UMAP" in f:
                del f["UMAP"]
            if "time" not in f:
                f.create_dataset("time", data=time.time() - start_time)
            else:
                f["time"][...] += time.time() - start_time

    Cell_BLAST.message.info("Saving results...")
    inferred_latent = model.inference(full_data_dict["x"])
    predicted_class = model.classify(full_data_dict["x"])
    Cell_BLAST.data.write_hybrid_path(inferred_latent, "%s//latent" % result_file)
    Cell_BLAST.data.write_hybrid_path(predicted_class, "%s//prediction" % result_file)


if __name__ == "__main__":
    main(parse_args())
    Cell_BLAST.message.info("Done!")
