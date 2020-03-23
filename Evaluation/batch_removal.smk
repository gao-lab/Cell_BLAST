configfile: "batch_removal_config.json"

rule batch_removal:
    input:
        expand(
            "../Results/batch_removal_{item}.{ext}",
            item=["2way", "map", "sas"],
            ext=config["plot_ext"]
        ),
        expand(
            "../Results/Cell_BLAST/{dataset}/dim_10_rmbatch0.01/seed_0/{vis}.{label}.{ext}",
            dataset=config["dataset"],
            vis=config["vis"],
            label=[config["label"], config["batch"]],
            ext=config["plot_ext"]
        ) if "Cell_BLAST" in config["latent_space_method"] else [],
        expand(
            "../Results/scVI/{dataset}/dim_5_rmbatchNA/seed_0/{vis}.{label}.{ext}",
            dataset=config["dataset"],
            vis=config["vis"],
            label=[config["label"], config["batch"]],
            ext=config["plot_ext"]
        ) if "scVI" in config["latent_space_method"] else [],
        "../Results/batch_removal_composition.xlsx"
    output:
        "../Results/.batch_removal_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule batch_removal_composition:
    input:
        data=expand("../Datasets/data/{ds}/data.h5", ds=config["dataset"]),
        script="batch_removal_composition.py"
    output:
        "../Results/batch_removal_composition.xlsx"
    threads: 1
    script:
        "batch_removal_composition.py"

rule batch_removal_plot:
    input:
        data="../Results/batch_removal.csv",
        palette="palette_method.json",
        script="batch_removal_plot.R"
    output:
        twoway="../Results/batch_removal_2way.{ext}".format(ext=config["plot_ext"]),
        cb_elbow="../Results/batch_removal_cb_elbow.{ext}".format(ext=config["plot_ext"]),
        twowayopt="../Results/batch_removal_2wayopt.{ext}".format(ext=config["plot_ext"]),
        map="../Results/batch_removal_map.{ext}".format(ext=config["plot_ext"]),
        sas="../Results/batch_removal_sas.{ext}".format(ext=config["plot_ext"])
    threads: 1
    script:
        "batch_removal_plot.R"

rule batch_removal_summary:
    input:
        ["{path}/metrics.json".format(path=item) for item in [
            "../Results/{method}/{dataset}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}".format(
                method=method,
                dataset=dataset,
                dimensionality=dimensionality,
                rmbatch="NA" if lambda_rmbatch_reg is None else lambda_rmbatch_reg,
                seed=seed
            ) for method in config["gene_space_method"] + config["latent_space_method"]
              for dataset in config["dataset"]
              for dimensionality in config[method]["dimensionality"]
              for lambda_rmbatch_reg in config[method]["lambda_rmbatch_reg"]
              for seed in (range(config[method]["seed"]) if isinstance(config[method]["seed"], int) else config[method]["seed"])
        ] if not os.path.exists("{path}/.blacklist".format(path=item))]
    output:
        "../Results/batch_removal.csv"
    params:
        pattern=lambda wildcards: "../Results/{method}/{dataset}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "summary.py"

rule batch_removal_metrics:
    input:
        data="../Datasets/data/{dataset}/data.h5",
        result=lambda wildcards: "../Results/{method}/{dataset}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/{filename}.h5".format(
            method=wildcards.method, dataset=wildcards.dataset,
            dimensionality=wildcards.dimensionality, rmbatch=wildcards.rmbatch, seed=wildcards.seed,
            filename="corrected" if wildcards.method in config["gene_space_method"] else "result"
        )
    output:
        "../Results/{method}/{dataset,.+\+.+}/dim_{dimensionality}_rmbatch{rmbatch}/seed_{seed}/metrics.json"
    params:
        slot=lambda wildcards: "exprs" if wildcards.method in config["gene_space_method"] else "latent"
    threads: 1
    script:
        "batch_removal_metrics.py"

rule merge_datasets:
    input:
        lambda wildcards: expand(
            "../Datasets/data/{item}/data.h5",
            item=wildcards.merged.split("+")
        )
    output:
        "../Datasets/data/{merged,.+\+.+}/data.h5"
    params:
        merge_uns_slots=["seurat_genes", "scmap_genes"],
        mapping=lambda wildcards: config[wildcards.merged]["merge_mapping"] \
            if "merge_mapping" in config[wildcards.merged] else None
    threads: 1
    script:
        "merge_datasets.py"

rule batch_removal_visualize:
    input:
        x="../Results/{method}/{dataset}/dim_{dimensionality}_rmbatch{other}/seed_{seed}/{vis}.h5",
        data="../Datasets/data/{dataset}/data.h5",
        script="visualize.py"
    output:
        "../Results/{method}/{dataset,.+\+.+}/dim_{dimensionality}_rmbatch{other}/seed_{seed}/{vis}.{label}.{ext}"
    params:
        shuffle=True,
        psize=lambda wildcards: config[wildcards.dataset]["psize"] \
            if "psize" in config[wildcards.dataset] else None,
        legend_size=lambda wildcards: config[wildcards.dataset]["legend_size"] \
            if wildcards.label == "cell_ontology_class" and "legend_size" in config[wildcards.dataset] else None,
        marker_scale=lambda wildcards: config[wildcards.dataset]["marker_scale"] \
            if wildcards.label == "cell_ontology_class" and "marker_scale" in config[wildcards.dataset] else None,
        label_spacing=lambda wildcards: config[wildcards.dataset]["label_spacing"] \
            if wildcards.label == "cell_ontology_class" and "label_spacing" in config[wildcards.dataset] else None,
        width=4,
        height=4,
        rasterized=True
    script:
        "visualize.py"

rule batch_removal_visualize_prep:
    input:
        "../Results/{method}/{dataset}/dim_{dimensionality}_rmbatch{other}/seed_{seed}/result.h5"
    output:
        "../Results/{method}/{dataset,.+\+.+}/dim_{dimensionality}_rmbatch{other}/seed_{seed}/{vis,tSNE|UMAP}.h5"
    script:
        "visualize_prep.py"

include: "batch_removal_worker.smk"