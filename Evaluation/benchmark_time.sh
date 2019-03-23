#!/bin/bash

set -e

export methods="scmap CellFishing.jl Cell_BLAST"
export genes="scanpy_genes"
export dims="10"
export timeout="8h"
export annotate="n_counts"

# Preparation
for method in ${methods}; do
    if [ ${method} = "Cell_BLAST" ]; then
        export benchmark="dimension_reduction"
        export datasets="1M_neurons"
        export epoch=100
        export patience=5
        export seeds="$(seq 0 15)"
        export n_jobs=2
        (cd ${method} && ./run_${method}.sh)
    fi
done

export benchmark="blast"
export scmap_thresholds="0.5"
export CellFishing_thresholds="110"
export Cell_BLAST_thresholds="0.05"
export seeds="$(seq 0 3)"
export n_jobs=1

for n_cell in 500 1000 5000 10000 50000 100000 500000 1000000; do
    export subsample_ref="${n_cell}"
    for method in ${methods}; do
        if [ ${method} = "Cell_BLAST" ] || [ ${method} == "CellFishing.jl" ]; then
            export datasets="1M_neurons:Marques"
            export align="no"
        else  # scmap
            export datasets="1M_neurons_half:Marques"
            if [ "${n_cell}" = "1000000" ]; then continue; fi
        fi
        (cd ${method} && ./run_${method}.sh)
    done
done

if [ ! -e "../Results/scmap/1M_neurons" ]; then
    ln -s 1M_neurons_half ../Results/scmap/1M_neurons
fi

export datasets="1M_neurons:Marques"
./benchmark_time.py
./benchmark_time.R
