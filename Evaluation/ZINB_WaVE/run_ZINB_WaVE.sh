#!/bin/bash

n_jobs=1  # Creates a lot of zombie processes, could reach system limit
seeds="$(seq 0 3)"

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v $1 -j${n_jobs} \
    test -e ../../Results/ZINB_WaVE/{1}/${genes}/dim_{3}/trial_{2}/result.h5 "||" \
    '(' \
        mkdir -p ../../Results/ZINB_WaVE/{1}/${genes}/dim_{3}/trial_{2} '&&' \
        timeout ${timeout} ./run_ZINB_WaVE.R \
            -i ../../Datasets/data/{1}/data.h5 \
            -o ../../Results/ZINB_WaVE/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -g ${genes} -d {3} -s {2} -j8 ${clean} '>' \
        ../../Results/ZINB_WaVE/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' \
    ')' \
::: ${datasets} ::: ${seeds} ::: ${dims}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code
