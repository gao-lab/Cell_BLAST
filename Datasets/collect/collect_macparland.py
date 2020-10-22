#! /usr/bin/env python
# by weil
# Sep 17, 2020

import pandas as pd
import numpy as np
import Cell_BLAST as cb
import scipy
import os
import scanpy as sc
from anndata import AnnData
from utils import construct_dataset

# expr matrix
expr_mat=pd.read_csv("../download/MacParland/GSE115469_Data.csv.gz", index_col=0)

# reshape to cell * gene
expr_mat=expr_mat.T

# cell meta
meta_df=pd.read_csv("../download/MacParland/Cell_clusterID_cycle.txt", sep="\t", index_col=0)
meta_df.columns=["donor", "barcode", "cluster", "cell_cycle"]
meta_df=meta_df.drop(columns="barcode")
meta_df["donor"]=meta_df["donor"].str.slice(0, 2)

# add donor meta
donor_meta=pd.read_csv("../download/MacParland/donor_annotation.csv")
meta_df["cell_id"]=meta_df.index
meta_df1=meta_df.merge(donor_meta, on="donor")

# add cluster annotation
cluster_annotation=pd.read_csv("../download/MacParland/cluster_annotation.csv")
meta_df2=meta_df1.merge(cluster_annotation, on="cluster")
meta_df2.index = meta_df2["cell_id"]
meta_df2=meta_df2.reindex(meta_df.index)
meta_df2=meta_df2.drop(columns="cell_id")

# datasets meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)
# cell ontology
cell_ontology = pd.read_csv("../cell_ontology/liver_cell_ontology.csv",
                            usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# gene_meta
gene_meta=pd.DataFrame(index=expr_mat.columns)

construct_dataset("../data/MacParland", expr_mat, meta_df2, gene_meta,
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology, min_mean=0.025, max_mean=3, min_disp=0.8)
