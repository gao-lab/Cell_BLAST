#! /usr/bin/env python
# by weil
# Sep 13, 2020


import pandas as pd
import numpy as np
import Cell_BLAST as cb
import os
import scanpy as sc
from utils import construct_dataset

#mature

expr_mature = sc.read_h5ad("../download/Stewart/Mature_Full_v2.1.h5ad")

meta_df_mature = expr_mature.obs.iloc[:, 0:4]
meta_df_mature.columns = ["cell_type1", "compartment", "cell_type0", "donor"]


# cell curation
# other donors have tumor
mature_mask = np.logical_or(np.logical_or(meta_df_mature["donor"] == "Teen Tx", meta_df_mature["donor"] == "TxK1"), 
                            meta_df_mature["donor"] == "TxK4")
meta_df_mature=meta_df_mature.loc[mature_mask, :]
expr_mat_mature=expr_mature.X[mature_mask.values, :]

# add donor meta data
donor_meta = {"age": ["12years9months", "53years", "72years"], "gender": ["F", "F", "M"], "donor": ["Teen Tx", "TxK1", "TxK4"]}
donor_meta = pd.DataFrame(data=donor_meta)
meta_df_mature["cell_id"] = meta_df_mature.index
meta_df_mature1 = meta_df_mature.merge(donor_meta, left_on="donor", right_on="donor")
meta_df_mature1.index = meta_df_mature1["cell_id"]
meta_df_mature1=meta_df_mature1.reindex(meta_df_mature["cell_id"])
meta_df_mature1=meta_df_mature1.drop(columns="cell_id")

# CL
cell_ontology_mature = pd.read_csv("../cell_ontology/kidney_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# dataset_meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)

# saving results
construct_dataset("../data/Stewart_Mature", expr_mat_mature, meta_df_mature1, expr_mature.var, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology_mature)


# fetal

expr_fetal = sc.read_h5ad("../download/Stewart/Fetal_full.h5ad")
meta_df_fetal = expr_fetal.obs.iloc[:, [1,2,3,4,6,7,8]]
meta_df_fetal.columns = ["sample", "donor", "gender", "age-PCW_weeks+days", "selection", "cell_type1", "compartment"]

# cell curation
# CNT/PC - proximal UB is ambiguous cell type
fetal_mask = meta_df_fetal["cell_type1"] != "CNT/PC - proximal UB"
meta_df_fetal=meta_df_fetal.loc[fetal_mask, :]
expr_mat_fetal=expr_fetal.X[fetal_mask.values, :]

# CL
cell_ontology_fetal = pd.read_csv("../cell_ontology/kidney_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# saving results
construct_dataset("../data/Stewart_Fetal", expr_mat_fetal, meta_df_fetal, expr_fetal.var, 
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology_fetal)
