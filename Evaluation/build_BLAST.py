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
    parser.add_argument("-m", "--models", dest="models", type=str, nargs="+")
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, required=True)
    parser.add_argument("-n", "--n-posterior", dest="n_posterior", type=int, default=50)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-j", "--n-jobs", dest="n_jobs", type=int, default=1)
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    cb.config.RANDOM_SEED = cmd_args.seed
    cb.config.N_JOBS = cmd_args.n_jobs
    return cmd_args


def main(cmd_args):

    cb.message.info("Reading data...")
    dataset = cb.data.ExprDataSet.read_dataset(cmd_args.ref)
    if cmd_args.clean:
        dataset = utils.clean_dataset(dataset, cmd_args.clean)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory()) \
        if cmd_args.device is None else cmd_args.device
    models = [cb.directi.DIRECTi.load(model) for model in cmd_args.models]

    cb.message.info("Building Cell BLAST index...")
    blast = cb.blast.BLAST(models, dataset, n_posterior=cmd_args.n_posterior)

    cb.message.info("Saving index...")
    blast.save(cmd_args.output_path)

    cb.message.info("Done!")


if __name__ == "__main__":
    main(parse_args())
