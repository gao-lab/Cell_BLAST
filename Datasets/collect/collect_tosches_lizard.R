#! /usr/bin/env Rscript
# by weil
# Jul 20,2018
# 9:24 PM

#This script collect data from reptile, lizard.

suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
load("../download/Reptile/lizard.all.cells.Robj", verbose = TRUE)
lizard.all <- UpdateSeuratObject(lizard.all)
meta_df <- lizard.all@meta.data
meta_df <- meta_df[ , c(4,5,13)]
colnames(meta_df) <- c("region", "donor", "cell_type1")
meta_df$gender <- "male"
meta_df$cell_type1 <- substring(meta_df$cell_type1, 3, 6)

expr_mat <- lizard.all@raw.data
expr_mat <- expr_mat[ , rownames(meta_df)]

gene_list <- list(
    author_genes = lizard.all@var.genes
)

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/reptile_brain_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset(save_dir = "../data/Tosches_lizard", expr_mat, meta_df, datasets_meta, cell_ontology, gene_list = gene_list, grouping = "donor")
message("Done!")
