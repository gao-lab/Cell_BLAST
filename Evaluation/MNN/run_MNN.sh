#!/bin/bash

n_jobs=1
seeds="0"  # MNN is deterministic, and we ignore PCA randomness here

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v $1 -j${n_jobs} \
    test -e ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/result.h5 '||' \
    '(' \
        mkdir -p ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2} '&&' \
        timeout ${timeout} ./run_MNN.R \
            -i ../../Datasets/data/{1}/data.h5 \
            -o ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -g ${genes} -b ${batch} -j8 ${clean} '>' \
        ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' '&&' \
        ../PCA/run_PCA.R \
            -i ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/result.h5//exprs \
            -o ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/result.h5 \
            -d {3} -s {2} '>>' \
        ../../Results/MNN/{1}/${genes}/dim_{3}/trial_{2}/output.txt '2>&1' \
    ')' \
::: ${datasets} ::: ${seeds} ::: ${dims}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code