#! /usr/bin/env Rscript
# by weil
# 2 Oct 2018
# 5:00 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Chen/chen.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))
meta_df$donor <- sapply(strsplit(rownames(meta_df), "_"), function(x){x[2]})
meta_df$region <- "Hypothalamus"

#clean cell type
mask <- meta_df$cell_type1 != "zothers"
expr_mat <- expr_mat[, mask]
meta_df <- meta_df[mask, c("cell_type1", "donor")]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mouse_brain_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Chen", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
