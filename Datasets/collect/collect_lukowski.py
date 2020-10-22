#! /usr/bin/env python
# by weil
# Sep 16, 2020

import pandas as pd
import numpy as np
import Cell_BLAST as cb
import scipy
import os
import scanpy as sc
from anndata import AnnData
from utils import construct_dataset

# expr_mat
# choose to use raw read counts, not processed data
expr_mat=pd.read_csv("../download/Lukowski/ae_exp_raw_all.tsv", sep="\t")
expr_mat=expr_mat.T

# meta_df
# use cell type annotation after batch effect correction by CCA
meta_df=pd.read_csv("../download/Lukowski/retina_wong_cellbc_cellid.csv", index_col=0)
meta_df=meta_df[["cell.id.cca"]]
meta_df.columns=["cell_type1"]

# cell curation based on cell_type1
cell_mask=meta_df["cell_type1"]!="Others CCA3"
cell_use=np.intersect1d(expr_mat.index, meta_df.index[cell_mask])

meta_df=meta_df.loc[cell_use]
expr_mat=expr_mat.loc[cell_use]

# datasets meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)
# cell ontology
cell_ontology = pd.read_csv("../cell_ontology/retina_cell_ontology.csv", 
                            usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# gene_meta
gene_meta=pd.DataFrame(index=expr_mat.columns)

construct_dataset("../data/Lukowski", expr_mat, meta_df, gene_meta, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology)
