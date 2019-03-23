#!/usr/bin/env python

import os
import pandas as pd
import tqdm
import Cell_BLAST as cb


def get_jobs():
    methods = os.environ["methods"].split()
    datasets = os.environ["datasets"].split()
    genes = os.environ["genes"]
    dim = os.environ["dims"]
    conf = "dim_%s" % dim


    for method in methods:
        for dataset in datasets:
            ref, query = dataset.split(":")
            pwd = os.path.join("../Results", method, ref, genes)
            if method == "Cell_BLAST":
                pwd = os.path.join(pwd, conf)
            trial_prefix = "blast_" if method == "Cell_BLAST" else "trial_"
            for trial in filter(
                lambda x, prefix=trial_prefix: x.startswith(prefix),
                os.listdir(pwd)
            ):
                for result_file in filter(
                    lambda x, query=query: x.startswith(query) and x.endswith(".h5"),
                    os.listdir(os.path.join(pwd, trial))
                ):
                    yield os.path.join(pwd, trial, result_file)


def do_job(pwd):
    pwd_split = pwd.split("/")
    if len(pwd_split) == 7:  # scmap and CellFishing.jl
        __, _, method, ref, genes, trial, result_file = pwd_split
    else:  # Cell_BLAST
        __, _, method, ref, genes, conf, trial, result_file = pwd_split
    query, ref_subsample_size = result_file.replace(".h5", "").split("_")
    time = cb.data.read_hybrid_path("//".join((pwd, "time")))
    return method, ref, trial, query, int(ref_subsample_size), time


def main():
    rs = []
    jobs = list(get_jobs())
    for pwd in tqdm.tqdm(jobs):
        rs.append(do_job(pwd))
    rs = list(zip(*rs))
    df = pd.DataFrame({
        "Method": rs[0],
        "Reference": rs[1],
        "Trial": rs[2],
        "Query": rs[3],
        "Reference size": rs[4],
        "Time per query (ms)": rs[5]
    })
    df.to_csv("../Results/benchmark_time.csv")


if __name__ == "__main__":
    main()
