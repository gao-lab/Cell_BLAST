import os

configfile: "dimension_reduction_config.json"

rule dimension_reduction:
    input:
        expand(
            "../Results/dimension_reduction_{item}.{ext}",
            item=["map", "optmap", "integrative"],
            ext=config["plot_ext"]
        ),
        expand(
            "../Results/Cell_BLAST/{dataset}/dim_10/seed_0/{vis}.{label}.{ext}",
            dataset=config["dataset"],
            vis=config["vis"],
            label=config["label"],
            ext=config["plot_ext"]
        ) if "Cell_BLAST" in config["method"] else []
    output:
        "../Results/.dimension_reduction_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule dimension_reduction_plot:
    input:
        data="../Results/dimension_reduction.csv",
        palette="palette_method.json",
        script="dimension_reduction_plot.R"
    output:
        map="../Results/dimension_reduction_map.{ext}".format(ext=config["plot_ext"]),
        optmap="../Results/dimension_reduction_optmap.{ext}".format(ext=config["plot_ext"]),
        integrative="../Results/dimension_reduction_integrative.{ext}".format(ext=config["plot_ext"])
    threads: 1
    script:
        "dimension_reduction_plot.R"

rule dimension_reduction_summary:
    input:
        [f"{path}/metrics.json" for path in [
            f"../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}"
            for method in config["method"]
            for dataset in config["dataset"]
            for dimensionality in (
                config[method]["dimensionality"]
                if method in config and "dimensionality" in config[method]
                else config["dimensionality"]
            )
            for seed in range(config["seed"])
        ] if not os.path.exists(f"{path}/.blacklist")]
    output:
        "../Results/dimension_reduction.csv"
    params:
        pattern=lambda wildcards: "../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "summary.py"

rule dimension_reduction_metrics:
    input:
        data="../Datasets/data/{dataset}/data.h5",
        result="../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    output:
        "../Results/{method}/{dataset,[^+]+}/dim_{dimensionality}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "dimension_reduction_metrics.py"

rule dimension_reduction_visualize:
    input:
        x="../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/{vis}.h5",
        data="../Datasets/data/{dataset}/data.h5",
        script="visualize.py"
    output:
        "../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/{vis}.{label}.{ext}"
    params:
        shuffle=True,
        psize=lambda wildcards: config[wildcards.dataset]["psize"] \
            if "psize" in config[wildcards.dataset] else None,
        legend_size=None,
        marker_scale=None,
        label_spacing=None,
        width=4,
        height=4,
        rasterized=True
    script:
        "visualize.py"

rule dimension_reduction_visualize_prep:
    input:
        "../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/result.h5"
    output:
        "../Results/{method}/{dataset}/dim_{dimensionality}/seed_{seed}/{vis,tSNE|UMAP}.h5"
    script:
        "visualize_prep.py"

include: "dimension_reduction_worker.smk"