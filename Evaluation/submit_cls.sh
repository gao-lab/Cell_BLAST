#!/bin/bash

snakemake -j500 --cluster-config cluster_cls.json --cluster "sbatch -J {cluster.jobname} -A {cluster.account} -p {cluster.partition} -q {cluster.qos} --no-requeue -N {cluster.n_node} -n {cluster.n_task} -c {cluster.n_cpu} {cluster.gres} -o {cluster.output} -e {cluster.error}" --js ./jobscript_cls.sh -pr
