#! /usr/bin/env python
# by weil
# Jun 29, 2020


import pandas as pd
import numpy as np
import Cell_BLAST as cb
import os
import scanpy as sc
from utils import construct_dataset

# Lung
expr_lung = sc.read_h5ad("../download/Madissoon/lung.cellxgene.h5ad")
expr_mat_lung = expr_lung.X
meta_df_lung = expr_lung.obs.iloc[:, [0,1,3,5,10]]
meta_df_lung.columns = ["donor", "cold_ischemic_time", "organ", "sample", "cell_type1"]

# add donor meta data
donor_meta = pd.read_excel("../download/Madissoon/madissoon_donor_meta.xlsx")
donor_meta=donor_meta.rename(columns={"ID": "donor", "DBD/DCD": "death_type", "Sex": "gender", "Age": "age"})
donor_meta=donor_meta.drop(columns="Tissues collected")
meta_df_lung["cell_id"] = meta_df_lung.index
meta_df_lung1 = meta_df_lung.merge(donor_meta, left_on="donor", right_on="donor")
meta_df_lung1.index = meta_df_lung1["cell_id"]
meta_df_lung1=meta_df_lung1.reindex(meta_df_lung["cell_id"])
meta_df_lung1=meta_df_lung1.drop(columns="cell_id")

# add CL
cell_ontology_lung = pd.read_csv("../cell_ontology/lung_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

gene_meta_lung = expr_lung.var.iloc[:, [0]]
gene_meta_lung.columns = ["gene_id"]

# dataset_meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)

# saving results
construct_dataset("../data/Madissoon_Lung", expr_mat_lung, meta_df_lung1, gene_meta_lung, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology_lung)


# Oesaphagus
expr_oesophagus = sc.read_h5ad("../download/Madissoon/oesophagus.cellxgene.h5ad")
expr_mat_oesophagus = expr_oesophagus.X
meta_df_oesophagus = expr_oesophagus.obs.iloc[:, [0,1,3,5,10]]
meta_df_oesophagus.columns = ["donor", "cold_ischemic_time", "organ", "sample", "cell_type1"]

# cell curation
# Mono_macro, NK_T_CD8_Cytotoxic are ambiguous cell types
mask_oesophagus = np.logical_and(meta_df_oesophagus["cell_type1"] != "NK_T_CD8_Cytotoxic", meta_df_oesophagus["cell_type1"] != "Mono_macro")
meta_df_oesophagus = meta_df_oesophagus.loc[mask_oesophagus, :]
expr_mat_oesophagus1 = expr_mat_oesophagus[mask_oesophagus.values, :]

meta_df_oesophagus["cell_id"] = meta_df_oesophagus.index
meta_df_oesophagus1 = meta_df_oesophagus.merge(donor_meta, left_on="donor", right_on="donor")
meta_df_oesophagus1.index = meta_df_oesophagus1["cell_id"]
meta_df_oesophagus1=meta_df_oesophagus1.reindex(meta_df_oesophagus["cell_id"])
meta_df_oesophagus1=meta_df_oesophagus1.drop(columns="cell_id")

# add CL
cell_ontology_oesophagus = pd.read_csv("../cell_ontology/oesophagus_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

gene_meta_oesophagus = expr_oesophagus.var.iloc[:, [0]]
gene_meta_oesophagus.columns = ["gene_id"]

# saving results
construct_dataset("../data/Madissoon_Oesophagus", expr_mat_oesophagus1, meta_df_oesophagus1, gene_meta_oesophagus, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology_oesophagus)


# spleen
expr_spleen = sc.read_h5ad("../download/Madissoon/spleen.cellxgene.h5ad")
expr_mat_spleen = expr_spleen.X
meta_df_spleen = expr_spleen.obs.iloc[:, [0,1,3,5,10]]
meta_df_spleen.columns = ["donor", "cold_ischemic_time", "organ", "sample", "cell_type1"]

# cell curation
# B_T_doublet, CD34_progenitor, Unknown are ambiguous cell types. T_CD8_MAIT is a new cell type not existed in CL, but retained here.
mask_spleen = np.logical_and(np.logical_and(meta_df_spleen["cell_type1"] != "B_T_doublet", meta_df_spleen["cell_type1"] != "CD34_progenitor"),\
                             meta_df_spleen["cell_type1"] != "Unknown")
meta_df_spleen = meta_df_spleen.loc[mask_spleen, :]
expr_mat_spleen1 = expr_mat_spleen[mask_spleen.values, :]

meta_df_spleen["cell_id"] = meta_df_spleen.index
meta_df_spleen1 = meta_df_spleen.merge(donor_meta, left_on="donor", right_on="donor")
meta_df_spleen1.index = meta_df_spleen1["cell_id"]
meta_df_spleen1=meta_df_spleen1.reindex(meta_df_spleen["cell_id"])
meta_df_spleen1=meta_df_spleen1.drop(columns="cell_id")

# add CL
cell_ontology_spleen = pd.read_csv("../cell_ontology/spleen_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# gene meta
gene_meta_spleen = expr_spleen.var.iloc[:, [0]]
gene_meta_spleen.columns = ["gene_id"]

# saving results
construct_dataset("../data/Madissoon_Spleen", expr_mat_spleen1, meta_df_spleen1, gene_meta_spleen, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology_spleen)
