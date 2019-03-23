#!/bin/bash

n_jobs=1  # Single run uses 20 cores
seeds="$(seq 0 3)"

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v $1 -j${n_jobs} \
    test -e ../../Results/ZIFA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 "||" \
    '(' \
        mkdir -p ../../Results/ZIFA/{1}/${genes}/dim_{3}/trial_{2} '&&' \
        timeout ${timeout} python -u ./run_ZIFA.py \
            -i ../../Datasets/data/{1}/data.h5 \
            -o ../../Results/ZIFA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -g ${genes} -l -d {3} -s {2} ${clean} '>' \
        ../../Results/ZIFA/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' \
    ')' \
::: ${datasets} ::: ${seeds} ::: ${dims}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code