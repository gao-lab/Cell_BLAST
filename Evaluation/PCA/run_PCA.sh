#!/bin/bash

n_jobs=4
seeds="$(seq 0 3)"  # PCA is very stable, doesn't need too many seeds

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v $1 -j${n_jobs} \
    test -e ../../Results/PCA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 "||" \
    '(' \
        mkdir -p ../../Results/PCA/{1}/${genes}/dim_{3}/trial_{2} '&&' \
        timeout ${timeout} ./run_PCA.R \
            -i ../../Datasets/data/{1}/data.h5 \
            -o ../../Results/PCA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -g ${genes} -l -d {3} -m irlba -s {2} ${clean} '>' \
        ../../Results/PCA/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' \
    ')' \
::: ${datasets} ::: ${seeds} ::: ${dims}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code