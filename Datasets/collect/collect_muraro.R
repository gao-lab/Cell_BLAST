#! /usr/bin/env Rscript
# by caozj
# 18 Dec 2017
# 4:11:20 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods


suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Muraro/muraro.rds")
count_mat <- as.matrix(normcounts(sce))
cdata <- as.data.frame(colData(sce))
rownames(count_mat) <- sapply(
    strsplit(rownames(count_mat), "__"), function(x) {
        x[1]
    }
)

message("Filtering...")

# Clean up cell type
mask <- cdata$cell_type1 != "unclear"
count_mat <- count_mat[, mask]
cdata <- cdata[mask, ]

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")

# Clean up ERCC
mask <- grepl("^ERCC-", rownames(count_mat))
count_mat <- count_mat[!mask, ]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Muraro", count_mat, cdata, datasets_meta, cell_ontology, grouping = "donor")
message("Done!")
