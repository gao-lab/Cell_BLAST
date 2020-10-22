#! /usr/bin/env Rscript
# by caozj
# 30 Oct 2017
# 10:45:24 PM
# This script organizes worm data into more easily accessible format
suppressPackageStartupMessages({
    library(Seurat)
    library(monocle)
})
source("../../Utilities/data.R", chdir = TRUE)
message("Reading data...")
load("../download/Worm/Cao_et_al_2017_vignette.RData")  # `cds` object
mat <- exprs(cds)
pd <- pData(cds)[, c(
    "cell.type", "region", "plate", "Cluster"
 )]
colnames(pd) <- c(
    "cell_type1", "region", "plate", "cluster"
)
pd$cell_type1[is.na(pd$cell_type1)] <- "NA"
pd$region[is.na(pd$region)] <- "NA"
mask <- !(pd$cell_type1 %in% c("Failed QC", "NA")) &
    !(pd$region %in% c("Failed QC", "NA"))
expr_mat <- mat[, mask]
meta_df <- pd[mask, ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/nematode_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]
#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Cao", expr_mat, meta_df, datasets_meta, cell_ontology)
cat("Done!\n")
