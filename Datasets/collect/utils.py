import pandas as pd
import numpy as np
import Cell_BLAST as cb
import os
import scanpy as sc
from scipy import sparse
from anndata import AnnData

def construct_dataset(output_dir, expr_mat, cell_meta, gene_meta, datasets_meta=None, cell_ontology=None,
                      gene_list=None, sparsity=True, min_mean=0.05, max_mean=3, min_disp=0.8,
                      compression="gzip", compression_opts=1, *args, **kwargs):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # sparse matrix or not
    if sparsity==True:
        expr_mat=sparse.csr_matrix(expr_mat)
    else:
        expr_mat=expr_mat.todense()
    
    # add dataset meta
    if not datasets_meta is None:
        dataset_name=os.path.basename(output_dir)
        cell_meta["organism"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organism"].item()
        cell_meta["dataset_name"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "dataset_name"].item()
        cell_meta["platform"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "platform"].item()
        cell_meta["organ"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organ"].item()
    
    # add CL
    if not cell_ontology is None:
        cell_meta["cell_id"] = cell_meta.index
        cell_meta1 = cell_meta.merge(cell_ontology, left_on="cell_type1", right_on="cell_type1")
        cell_meta1.index = cell_meta1["cell_id"]
        cell_meta1=cell_meta1.reindex(cell_meta["cell_id"])
        cell_meta=cell_meta1.drop(columns="cell_id")
  
    
    # use scanpy for normalization and variable gene selection
    print("Creating AnnData...")
    if isinstance(expr_mat, pd.DataFrame):
        adata=AnnData(X=expr_mat.values, obs=cell_meta, var=gene_meta)
    else:
        adata=AnnData(X=expr_mat, obs=cell_meta, var=gene_meta)
    print("Selecting scanpy genes...")    
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, inplace=True,
                                min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, *args, **kwargs)
    print(np.sum(adata.var["highly_variable"]), "scanpy genes selected!")
    # plot
    sc.pl.highly_variable_genes(adata, save=".pdf")
    import shutil
    shutil.move("./figures/filter_genes_dispersion.pdf", os.path.join(output_dir, "scanpy_genes.pdf"))

### change to anndata will use below codes:
#     adata.X = adata.raw.X
#     adata.raw = None
#     print("Saving results...")
#     adata.write(os.path.join(output_dir, "data.h5ad"), compression=compression, compression_opts=compression_opts)
    
    scanpy_genes = adata.var_names[adata.var["highly_variable"]] # to be deleted
    if gene_list is None:
        gene_list={}
    gene_list["scanpy_genes"]=np.array(scanpy_genes)
    
    # saving results
    print("Saving results...")
    dataset = cb.data.ExprDataSet(
        expr_mat, cell_meta, gene_meta, gene_list
    )
    
    # write dataset
    dataset.write_dataset(os.path.join(output_dir, "data.h5"))
    
    print("Done!")