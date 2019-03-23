#! /usr/bin/env Rscript
# by weil
# 4 Oct 2018
# 10:40 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Campbell/campbell.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))

mask <- meta_df$X7.clust_all != "miss"
expr_mat <- expr_mat[, mask]
meta_df <- meta_df[mask, c("X2.group", "X3.batches", "X4.sex", "X7.clust_all", "cell_type1" )]
colnames(meta_df) <- c("diet", "batch", "gender",  "cluster", "cell_type2")

cluster_annotation <- read.csv("../download/Hemberg/Campbell/Campbell_cluster_annotation.csv", header = TRUE)
meta_df$cellID <- rownames(meta_df)
meta_df <- merge(meta_df, cluster_annotation, by = "cluster")
rownames(meta_df) <- meta_df$cellID
meta_df$cellID <- NULL
meta_df$cluster <- NULL
meta_df$region <- "Hypothalamus and median eminence"
#clean ERCC
expr_mat <- expr_mat[!grepl("ERCC-", rownames(expr_mat)),]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mouse_brain_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Campbell", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")

