#!/usr/bin/env python

"""
This script summarizes performance of tests for simulated duplicated genes problem
"""

import sys
import os
import pandas as pd
sys.path.append("../Utilities")
import metrics
import data

y = data.read_hybrid_path("../Datasets/data/Baron_human+Segerstolpe/data.h5//obs/cell_ontology_class")
z = data.read_hybrid_path("../Datasets/data/Baron_human+Segerstolpe/data.h5//obs/study")

pwd = "../Results/Cell_BLAST"
datasets = os.listdir(pwd)

df_dataset, df_map, df_seurat_score, df_trial = [], [], [], []
for dataset in datasets:
    pwd = os.path.join("../Results/Cell_BLAST", dataset, "seurat_genes/dim_10")
    for trial in os.listdir(pwd):
        result_file = os.path.join(pwd, trial, "result.h5")
        x = data.read_hybrid_path("//".join([result_file, "latent"]))
        mean_average_precision = metrics.mean_average_precision_from_latent(x, y, n_jobs=8)
        seurat_alignment_score = metrics.seurat_alignment_score(x, z, n=10, n_jobs=8)
        df_map.append(mean_average_precision)
        df_seurat_score.append(seurat_alignment_score)
        df_dataset.append(dataset)
        df_trial.append(trial)

df = pd.DataFrame({
    "Dataset": df_dataset,
    "Trial": df_trial,
    "Mean Average Precision": df_map,
    "Seurat Alignment Score": df_seurat_score
})
df.to_csv("../Results/Cell_BLAST/test_for_simulated_dupgene.csv", index=False)

