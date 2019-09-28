rule batch_removal_pca:  # Negative control
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/PCA/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/result.h5"
    log:
        "../Results/PCA/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/log.txt"
    params:
        blacklist="../Results/PCA/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    params:
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_PCA.R -i {input} -o {output} -g {config[genes]} "
        "-l -d {wildcards.dimensionality} -m irlba -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_cb:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/Cell_BLAST/{dataset,.+\+.+}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/result.h5"
    log:
        "../Results/Cell_BLAST/{dataset}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/Cell_BLAST/{dataset}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DIRECTi.py -i {input} -o {output} -g {config[genes]} "
        "-b {config[batch]} --lambda-rmbatch-reg {wildcards.rmbatch} --prob-module NB -l {wildcards.dimensionality} "
        "-c {config[Cell_BLAST][n_cluster]} --no-normalize -s {wildcards.seed} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_combat:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/ComBat/{dataset,.+\+.+}/corrected.h5"
    log:
        "../Results/Combat/{dataset}/log.txt"
    params:
        blacklist="../Results/ComBat/{dataset}/corrected.h5"  # This will cause batch_removal_mnn_pca to tail, effectively blacklisting it
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_ComBat.R -i {input} -o {output} "
        "-g {config[genes]} -b {config[batch]} -j {threads} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_mnn:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/MNN/{dataset,.+\+.+}/corrected.h5"
    log:
        "../Results/MNN/{dataset}/log.txt"
    params:
        blacklist="../Results/MNN/{dataset}/corrected.h5"  # This will cause batch_removal_mnn_pca to tail, effectively blacklisting it
    threads: 8
    shell:
        "timeout {config[timeout]} Rscript run_MNN.R -i {input} -o {output} "
        "-g {config[genes]} -b {config[batch]} -j {threads} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_followup_pca:
    input:
        "../Results/{method}/{dataset}/corrected.h5"
    output:
        "../Results/{method,MNN|ComBat}/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/result.h5"
    log:
        "../Results/{method}/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/log.txt"
    params:
        blacklist="../Results/{method}/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_PCA.R -i {input}//exprs -o {output} "
        "-d {wildcards.dimensionality} -m irlba -s {wildcards.seed} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_cca:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/CCA/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/result.h5"
    log:
        "../Results/CCA/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/log.txt"
    params:
        blacklist="../Results/CCA/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript run_CCA.R -i {input} -o {output} "
        "-g {config[genes]} -b {config[batch]} -d {wildcards.dimensionality} "
        "-s {wildcards.seed} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_cca_anchor:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/CCA_anchor/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/corrected.h5"
    log:
        "../Results/CCA_anchor/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/log.txt"
    params:
        blacklist="../Results/CCA_anchor/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript run_CCA_anchor.R -i {input} -o {output} "
        "-g {config[genes]} -b {config[batch]} -d {wildcards.dimensionality} "
        "-s {wildcards.seed} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_cca_anchor_pca:
    input:
        "../Results/CCA_anchor/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/corrected.h5"
    output:
        "../Results/CCA_anchor/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/result.h5"
    log:
        "../Results/CCA_anchor/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/pca_log.txt"
    params:
        blacklist="../Results/CCA_anchor/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_PCA.R -i {input}//exprs -o {output} "
        "-d {wildcards.dimensionality} -m irlba -s {wildcards.seed} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule batch_removal_scvi:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/scVI/{dataset,.+\+.+}/dim_{dimensionality}_rmbatchNA/seed_{seed}/result.h5"
    log:
        "../Results/scVI/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/log.txt"
    params:
        blacklist = "../Results/scVI/{dataset}/dim_{dimensionality}_rmbatchNA/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu = 1
    shell:
        "timeout {config[timeout]} python -u run_scVI.py "
        "-i {input} -o {output} -g {config[genes]} -b {config[batch]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} --clean {config[label]} "
        "> {log} 2>&1 || touch {params.blacklist}"
