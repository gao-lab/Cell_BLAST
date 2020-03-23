rule dimension_reduction_pca:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/PCA/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/PCA/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/PCA/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_PCA.R -i {input} -o {output} -g {config[genes]} "
        "-l -d {wildcards.dimensionality} -m irlba -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_tsne:
    input:
        lambda wildcards: f"../Results/PCA/{wildcards.dataset}/dim_{config['tSNE']['pca']}/seed_{wildcards.seed}/result.h5"
    output:
        "../Results/tSNE/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/tSNE/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/tSNE/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} Rscript run_tSNE.R -i {input}//latent -o {output} "
        "-d {wildcards.dimensionality} -j {threads} -s {wildcards.seed} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_umap:
    input:
        lambda wildcards: f"../Results/PCA/{wildcards.dataset}/dim_{config['UMAP']['pca']}/seed_{wildcards.seed}/result.h5"
    output:
        "../Results/UMAP/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/UMAP/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/UMAP/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    shell:
        "timeout {config[timeout]} python -u run_UMAP.py -i {input}//latent -o {output} "
        "-d {wildcards.dimensionality} -s {wildcards.seed} "
        "> {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_zifa:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/ZIFA/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/ZIFA/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/ZIFA/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 20
    shell:
        "timeout {config[timeout]} python -u run_ZIFA.py -i {input} -o {output} -g {config[genes]} "
        "-d {wildcards.dimensionality} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_zinbwave:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/ZINB_WaVE/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/ZINB_WaVE/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/ZINB_WaVE/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 8
    shell:
        "timeout {config[timeout]} Rscript run_ZINB_WaVE.R -i {input} -o {output} -g {config[genes]} "
        "-d {wildcards.dimensionality} -s {wildcards.seed} -j {threads} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_dhaka:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/Dhaka/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/Dhaka/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/Dhaka/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_Dhaka.py -i {input} -o {output} -g {config[genes]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_dca:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/DCA/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/DCA/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/DCA/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DCA.py -i {input} -o {output} -g {config[genes]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} -t {threads} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_dca_modpp:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/DCA_modpp/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/DCA_modpp/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/DCA_modpp/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DCA_modpp.py -i {input} -o {output} -g {config[genes]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} -t {threads} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_scvi:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/scVI/{dataset,[^+]+}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/scVI/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/scVI/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_scVI.py -i {input} -o {output} -g {config[genes]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_scscope:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/scScope/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/scScope/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/scScope/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_scScope.py -i {input} -o {output} -g {config[genes]} "
        "--n-latent {wildcards.dimensionality} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"

rule dimension_reduction_cb:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/Cell_BLAST/{dataset,[^+]+}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/Cell_BLAST/{dataset}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/Cell_BLAST/{dataset}/dim_{dimensionality}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DIRECTi.py -i {input} -o {output} -g {config[genes]} "
        "--prob-module NB -l {wildcards.dimensionality} -c {config[Cell_BLAST][n_cluster]} "
        "--no-normalize -s {wildcards.seed} --clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"
