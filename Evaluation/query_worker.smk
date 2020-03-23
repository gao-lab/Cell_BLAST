rule query_scmap:
    input:
        ref=lambda wildcards: expand(
            "../Datasets/data/{ref}/data.h5",
            ref=config[wildcards.group]["ref"]
        ),
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/scmap/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/scmap/{group}/seed_{seed}/{query}/log.txt"
    params:
        ref_names=lambda wildcards: config[wildcards.group]["ref"]
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_scmap.R "
        "-r {input.ref} -n {params.ref_names} "
        "-g scmap_genes -c {config[label]} --threshold {config[scmap][threshold]} "
        "-q {input.query} -o {output} -s {wildcards.seed} --shuffle-genes "
        "--clean {config[label]} > {log} 2>&1"

rule query_cf:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref="+".join(config[wildcards.group]["ref"])
        ),
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/CellFishing.jl/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/CellFishing.jl/{group}/seed_{seed}/{query}/log.txt"
    params:
        thresholds=",".join([
            str(item) for item in config["CellFishing.jl"]["threshold"]])
    threads: 4
    shell:
        "timeout {config[timeout]} julia run_CellFishing.jl.jl "
        "--annotation={config[label]} --cutoff={params.thresholds} "
        "--gene=cf_genes --seed={wildcards.seed} --clean={config[label]} "
        "{input.ref} {input.query} {output} > {log} 2>&1"

rule query_cb_build:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref="+".join(config[wildcards.group]["ref"])
        ),
        models=lambda wildcards: expand(
            "../Results/Cell_BLAST/{ref}/dim_{dimensionality}{rmbatch}/seed_{seed}/result.h5",
            ref="+".join(config[wildcards.group]["ref"]),
            dimensionality=config["Cell_BLAST"]["dimensionality"],
            rmbatch="" if len(config[wildcards.group]["ref"]) == 1
                else "_rmbatch" + str(config["Cell_BLAST"]["lambda_rmbatch_reg"]),
            seed=range(
                int(wildcards.seed) * config["Cell_BLAST"]["n_models"],
                (int(wildcards.seed) + 1) * config["Cell_BLAST"]["n_models"]
            )
        )  # Actually model results
    output:
        directory("../Results/Cell_BLAST/{group}/seed_{seed}/blast")
    log:
        "../Results/Cell_BLAST/{group}/seed_{seed}/log.txt"
    params:
        models=lambda wildcards, input: [os.path.dirname(item) for item in input.models]
    threads: config["Cell_BLAST"]["n_models"]
    shell:
        "timeout {config[timeout]} python -u build_BLAST.py "
        "-r {input.ref} -m {params.models} -o {output} -j {threads} "
        "-s {wildcards.seed} --clean {config[label]} > {log} 2>&1"

rule query_cb:
    input:
        blast="../Results/Cell_BLAST/{group}/seed_{seed}/blast",
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/Cell_BLAST/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/Cell_BLAST/{group}/seed_{seed}/{query}/log.txt"
    threads: config["Cell_BLAST"]["n_models"]
    shell:
        "timeout {config[timeout]} python -u run_BLAST.py "
        "-i {input.blast} -o {output} -q {input.query} -a {config[label]} "
        "-c {config[Cell_BLAST][threshold]} -j {threads} "
        "-s {wildcards.seed} --clean {config[label]} > {log} 2>&1"

rule query_scanvi_train:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref="+".join(config[wildcards.group]["ref"])
        )
    output:
        "../Results/{scanvi,scANVI.*}/{group}/dim_{dimensionality}/seed_{seed}/result.h5"
    log:
        "../Results/{scanvi}/{group}/dim_{dimensionality}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/{scanvi}/{group}/dim_{dimensionality}/seed_{seed}/.blacklist",
        normalized=lambda wildcards: "-n" if wildcards.scanvi == "scANVI_normalized" else ""
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_scVI.py -i {input} -o {output} "
        "-g {config[genes]} --n-latent {wildcards.dimensionality} "
        "--supervision {config[label]} --label-fraction 1.0 --clean {config[label]} {params.normalized} "
        "-s {wildcards.seed} > {log} 2>&1 || touch {params.blacklist}"

rule query_scanvi:
    input:
        model=lambda wildcards: "../Results/{scanvi}/{group}/dim_{dimensionality}/seed_{seed}/result.h5".format(
            scanvi=wildcards.scanvi,
            group=wildcards.group,
            dimensionality=config["scANVI"]["dimensionality"],
            seed=wildcards.seed
        ),
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/{scanvi,scANVI.*}/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/{scanvi}/{group}/seed_{seed}/{query}/log.txt"
    params:
        model=lambda wildcards, input: os.path.dirname(input.model),
        normalized=lambda wildcards: "-n" if wildcards.scanvi == "scANVI_normalized" else "",
        threshold=lambda wildcards: config[wildcards.scanvi]["threshold"]
    threads: 4
    shell:
        "timeout {config[timeout]} python -u run_scANVI_query.py -m {params.model} "
        "-q {input.query} -c {params.threshold} -o {output} {params.normalized} "
        "--clean {config[label]} > {log} 2>&1"

rule query_dca:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref="+".join(config[wildcards.group]["ref"])
        ),
        model=lambda wildcards: "../Results/DCA_modpp/{ref}/dim_{dimensionality}/seed_{seed}/result.h5".format(
            ref="+".join(config[wildcards.group]["ref"]),
            dimensionality=config["DCA_modpp"]["dimensionality"],
            seed=wildcards.seed
        ),  # Actually model results
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/DCA_modpp/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/DCA_modpp/{group}/seed_{seed}/{query}/log.txt"
    params:
        model=lambda wildcards, input: os.path.dirname(input.model)
    threads: 1
    shell:
        "timeout {config[timeout]} python run_DCA_query.py "
        "-m {params.model} -r {input.ref} -q {input.query} -o {output} "
        "-a {config[label]} -c {config[DCA_modpp][threshold]} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1"

rule query_pca:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref="+".join(config[wildcards.group]["ref"])
        ),
        model=lambda wildcards: "../Results/PCA/{ref}/dim_{dimensionality}/seed_{seed}/result.h5".format(
            ref="+".join(config[wildcards.group]["ref"]),
            dimensionality=config["PCA"]["dimensionality"],
            seed=wildcards.seed
        ),
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/PCA/{group}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/PCA/{group}/seed_{seed}/{query}/log.txt"
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_PCA_query.R "
        "-m {input.model} -r {input.ref} -q {input.query} -o {output} "
        "-a {config[label]} -c {config[PCA][threshold]} -s {wildcards.seed} "
        "--clean {config[label]} > {log} 2>&1"
