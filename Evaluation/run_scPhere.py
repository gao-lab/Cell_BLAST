#! /usr/bin/env python
# by caozj
# May 1, 2020
# 10:33:52 PM

try:
    import scphere.model.vae
    import scphere.util.trainer
except ModuleNotFoundError:
    import sys
    import subprocess
    subprocess.Popen(
        "source <($CONDA_EXE shell.bash hook 2> /dev/null) && "
        "conda activate scphere && "
        "python -u " + " ".join(sys.argv),
        shell=True, executable="/bin/bash"
    ).wait()
    sys.exit(0)

import os
import argparse
import time
import numpy as np
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
    parser.add_argument("-b", "--batch-effect", dest="batch_effect", type=str, nargs="*")
    parser.add_argument("-d", "--dim", dest="dim", type=int, default=2)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    cmd_args.output_path = os.path.dirname(cmd_args.output)
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    return cmd_args


def main(cmd_args):
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.input, sparsify=True)
    if cmd_args.clean is not None:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)
    dataset = dataset[:, dataset.uns[cmd_args.genes]]
    dataset.exprs = cb.utils.densify(dataset.exprs)
    if cmd_args.batch_effect:
        batch_id = [cb.utils.encode_integer(
            dataset.obs[batch_effect].astype(object).fillna("NA")
        )[0] for batch_effect in cmd_args.batch_effect]
        n_batch = [np.unique(item).size for item in batch_id]
        batch_id = np.stack(batch_id, axis=1)
        if len(cmd_args.batch_effect) == 1:
            n_batch = n_batch[0]
            batch_id = batch_id[:, 0]
    else:
        n_batch = 0
        batch_id = np.zeros(dataset.shape[0]) * -1

    start_time = time.time()
    model = scphere.model.vae.SCPHERE(
        n_gene=dataset.shape[1], z_dim=cmd_args.dim,
        latent_dist="vmf", observation_dist="nb", seed=cmd_args.seed,
        n_batch=n_batch
    )
    trainer = scphere.util.trainer.Trainer(
        x=dataset.exprs, model=model,
        mb_size=128, learning_rate=0.001, max_epoch=250,
        batch_id=batch_id
    )
    trainer.train()

    latent = model.encode(dataset.exprs, batch_id)
    cb.data.write_hybrid_path(
        time.time() - start_time,
        "//".join([cmd_args.output, "time"])
    )
    cb.data.write_hybrid_path(latent, "//".join([cmd_args.output, "latent"]))

    model.save_sess(os.path.join(cmd_args.output_path, "model"))


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
