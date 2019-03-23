#! /usr/bin/env Rscript
# by weil
# Jan 3, 2018
# 02:30 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- readRDS("../download/Bach/ExpressionList_QC_norm_clustered_clean.rds")
counts <- expr_mat$counts
cell_meta <- expr_mat$phenoData
gene_meta <- expr_mat$featureData

#cell_meta
mask <- cell_meta$PassAll == TRUE
cell_meta <- cell_meta[mask, c("barcode", "Condition", "UmiSums", "prcntMito", "sample", "SuperCluster")]
rownames(cell_meta) <- cell_meta$barcode
cell_meta$barcode <- NULL
colnames(cell_meta) <- c("lifestage", "umisums", "percent.mito", "donor", "cell_type1")
cell_meta$region = "epithelial cells"


#gene_meta
dup <- duplicated(gene_meta$symbol)
counts <- counts[!dup, mask]
gene_meta <- gene_meta[!dup, ]
rownames(gene_meta) <- gene_meta$symbol
rownames(counts) <- gene_meta$symbol

#clean cell type
cell_mask <- !is.na(cell_meta$cell_type1)
cell_meta <- cell_meta[cell_mask, ]
counts <- counts[, cell_mask]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mammary_gland_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Bach", counts, cell_meta = cell_meta, gene_meta = gene_meta,
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology)
message("Done!")
