#!/bin/bash

device_rpl="{d} 1 @pool = ($(cat ../devices.txt)); \$_ = \$pool[(\$job->seq() - 1) % (\$#pool + 1)]"
if [ ! -z "${clean}" ]; then clean="--clean ${clean}"; fi

if [ "${benchmark}" = "blast" ]; then
    if [ -z "${seeds}" ]; then seeds="$(seq 0 3)"; fi
    if [ -z "${n_jobs}" ]; then n_jobs=8; fi
    if [ "${align}" = "yes" ]; then
        align_opt="--align"
        align_res="_aligned"
    else  # "no"
        align_opt=""
        align_res=""
    fi
    if [ ! -z "${subsample_ref}" ]; then
        subsample_suffix="_${subsample_ref}"
        subsample_ref="--subsample-ref ${subsample_ref}"
    fi

    parallel -v --dryrun -j1 -k \
        --rpl "{ref} 1 \$_ = (split(/:/, \$_))[0]; \$rmbatch = (\$_ =~ m/\\+/)" \
        --rpl "{sup} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? (('${label_priority}' eq '') ? '_sup_' . \$_ : '_SUP_' . \$_) : ''" \
        --rpl "{rmbatch} 4 \$_ = (\$rmbatch && \$_ ne '') ? '_rmbatch_' . \$_ : ''" \
    test -e ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/index.pkz '|| (' \
        mkdir -p ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2} '&&' \
        python -u ./build_blast_index.py \
            -r ../../Datasets/data/{ref}/data.h5 \
            -m ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch} \
            -o ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2} \
            -t 4 -j 1 -s {2} -d {device} ${clean} '>' \
        ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/build_blast_index_output.txt '2>&1' \
    ')' \
    ::: ${datasets} ::: ${seeds} ::: ${label_fractions} ::: ${lambda_rmbatch_regs} ::: ${dims} | \
    tr -d "'" | tr -s " " | sort | uniq | \
    parallel -j1 -k --rpl "${device_rpl}" echo {} '|' sed 's/{device}/{d}/g' | \
    parallel -v -j${n_jobs} $1 {}

    parallel -v --dryrun -j1 -k \
        --rpl "{ref} 1 \$_ = (split(/:/, \$_))[0]; \$rmbatch = (\$_ =~ m/\\+/)" \
        --rpl "{query} 1 \$_ = (split(/:/, \$_))[1]" \
        --rpl "{sup} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? (('${label_priority}' eq '') ? '_sup_' . \$_ : '_SUP_' . \$_) : ''" \
        --rpl "{rmbatch} 4 \$_ = (\$rmbatch && \$_ ne '') ? '_rmbatch_' . \$_ : ''" \
    test -e ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/{query}${subsample_suffix}${align_res}.h5 '|| (' \
        test -e ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/index.pkz '&&' \
        python -u ./run_blast.py \
            -i ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2} \
            -o ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/{query}${subsample_suffix}${align_res}.h5 \
            -q ../../Datasets/data/{query}/data.h5 -a ${annotate} -c ${Cell_BLAST_thresholds} \
            ${align_opt} ${subsample_ref} -j 4 -s {2} -d {device} ${clean} '>' \
        ../../Results/Cell_BLAST/{ref}/${genes}/dim_{5}{sup}{rmbatch}/blast_{2}/{query}${subsample_suffix}${align_res}_output.txt '2>&1' \
    ')' \
    ::: ${datasets} ::: ${seeds} ::: ${label_fractions} ::: ${lambda_rmbatch_regs} ::: ${dims} | \
    tr -d "'" | tr -s " " | sort | uniq | \
    parallel -j1 -k --rpl "${device_rpl}" echo {} '|' sed 's/{device}/{d}/g' | \
    parallel -v -j${n_jobs} $1 {}
else
    if [ -z "${seeds}" ]; then seeds="$(seq 0 15)"; fi
    if [ -z "${n_jobs}" ]; then n_jobs=8; fi
    if [ -z "${epoch}" ]; then epoch=1000; fi
    if [ -z "${patience}" ]; then patience=30; fi
    parallel -v --dryrun -j1 -k \
        --rpl "{sup} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? (('${label_priority}' eq '') ? '_sup_' . \$_ : '_SUP_' . \$_) : ''" \
        --rpl "{rmbatch} 4 \$_ = (\$rmbatch && \$_ ne '') ? '_rmbatch_' . \$_ : ''" \
        --rpl "{:cluster} 1 \$_ = ('${supervision}' eq '') ? '-c 20' : ''" \
        --rpl "{:batch} 1 \$rmbatch = (\$_ =~ m/\\+/); \$_ = \$rmbatch ? '-b ${batch}' : ''" \
        --rpl "{:lambda_rmbatch_reg} 4 \$_ = (\$rmbatch && \$_ ne '') ? '--lambda-rmbatch-reg ' . \$_ : ''" \
        --rpl "{:supervision} 1 \$_ = ('${supervision}' ne '') ? '--supervision ${supervision}' : ''" \
        --rpl "{:label_priority} 1 \$_ = ('${supervision}' ne '' && '${label_priority}' ne '') ? '--label-priority ${label_priority}' : ''" \
        --rpl "{:label_fraction} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? '--label-fraction ' . \$_ : ''" \
    test -e ../../Results/Cell_BLAST/{1}/${genes}/dim_{5}{sup}{rmbatch}/trial_{2}/result.h5 '|| (' \
        mkdir -p ../../Results/Cell_BLAST/{1}/${genes}/dim_{5}{sup}{rmbatch}/trial_{2} '&&' \
        timeout ${timeout} python -u ./run_model.py \
            -i ../../Datasets/data/{1}/data.h5 -g ${genes} \
            -o ../../Results/Cell_BLAST/{1}/${genes}/dim_{5}{sup}{rmbatch}/trial_{2} \
            --prob-module NB -l {5} {:cluster} \
            {:supervision} {:label_priority} {:label_fraction} \
            {:batch} {:lambda_rmbatch_reg} \
            --epoch ${epoch} --patience ${patience} -s {2} -d {device} ${clean} '>' \
        ../../Results/Cell_BLAST/{1}/${genes}/dim_{5}{sup}{rmbatch}/trial_{2}/output.txt '2>&1' \
    ')' \
    ::: ${datasets} ::: ${seeds} ::: ${label_fractions} ::: ${lambda_rmbatch_regs} ::: ${dims} | \
    tr -d "'" | tr -s " " | sort | uniq | \
    parallel -j1 -k --rpl "${device_rpl}" echo {} '|' sed 's/{device}/{d}/g' | \
    parallel -v -j${n_jobs} $1 {}
fi

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code