#!/bin/bash

n_jobs=4
seeds="0 1 2 3"

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v $1 -j${n_jobs} \
    test -e ../../Results/CCA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 '||' \
    '(' \
        mkdir -p ../../Results/CCA/{1}/${genes}/dim_{3}/trial_{2} '&&' \
        timeout ${timeout} ./run_CCA.R \
            -i ../../Datasets/data/{1}/data.h5 \
            -o ../../Results/CCA/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -g ${genes} -b ${batch} -d {3} ${clean} '>' \
        ../../Results/CCA/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' \
    ')' \
::: ${datasets} ::: ${seeds} ::: ${dims}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code
