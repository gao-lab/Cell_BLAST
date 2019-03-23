#!/bin/bash

if [ -z "${n_jobs}" ]; then n_jobs=4; fi
if [ -z "${seeds}" ]; then seeds="$(seq 0 3)"; fi
if [ ! -z "${clean}" ]; then clean="--clean=${clean}"; fi
if [ ! -z "${subsample_ref}" ]; then
    subsample_suffix="_${subsample_ref}"
    subsample_ref="--subsample-ref=${subsample_ref}"
fi

parallel -j1 echo {1}:{2} ::: ${datasets} ::: ${seeds} | \
parallel -v --colsep ':' $1 -j${n_jobs} \
    test -e ../../Results/CellFishing.jl/{1}/${genes}/trial_{3}/{2}${subsample_suffix}.h5 '||' \
    '(' \
        mkdir -p ../../Results/CellFishing.jl/{1}/${genes}/trial_{3} '&&' \
        ./run_CellFishing.jl.jl \
            --annotation=${annotate} --cutoff=${CellFishing_thresholds} \
            --genes=${genes} --seed={3} ${clean} ${subsample_ref} \
            ../../Datasets/data/{1}/data.h5 ../../Datasets/data/{2}/data.h5 \
            ../../Results/CellFishing.jl/{1}/${genes}/trial_{3}/{2}${subsample_suffix}.h5 '>' \
        ../../Results/CellFishing.jl/{1}/${genes}/trial_{3}/{2}${subsample_suffix}_output.txt '2>&1' \
    ')'
