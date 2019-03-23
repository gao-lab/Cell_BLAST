#!/usr/bin/env python

import os
import sys
import argparse

import Cell_BLAST as cb
sys.path.insert(0, "..")
import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", dest="ref", type=str, required=True)
    parser.add_argument("-m", "--model", dest="model", type=str, required=True)
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, required=True)
    parser.add_argument("-n", "--n-posterior", dest="n_posterior", type=int, default=50)
    parser.add_argument("-t", "--n-trials", dest="n_trials", type=int, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-j", "--n-jobs", dest="n_jobs", type=int, default=1)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    cb.config.RANDOM_SEED = cmd_args.seed
    cb.config.N_JOBS = cmd_args.n_jobs
    return cmd_args


def main(cmd_args):

    # models = random.sample([
    #     item for item in os.listdir(cmd_args.model) if item.startswith("trial_")
    # ], cmd_args.n_trials)
    models = ["trial_%d" % trial for trial in range(
        cmd_args.n_trials * cmd_args.seed,
        cmd_args.n_trials * (cmd_args.seed + 1)
    )]
    models = [cb.directi.DIRECTi.load(
        os.path.join(cmd_args.model, model)
    ) for model in models]

    cb.message.info("Reading data...")
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.ref).normalize()
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)

    cb.message.info("Building cell BLAST index...")
    blast = cb.blast.BLAST(
        models, dataset, keep_exprs=True, n_posterior=cmd_args.n_posterior
    ).build_empirical()

    cb.message.info("Saving index...")
    blast.save(cmd_args.output_path)

    cb.message.info("Done!")


if __name__ == "__main__":
    main(parse_args())
