#! /usr/bin/env Rscript
# by caozj
# 18 Dec 2017
# 4:11:20 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods


suppressPackageStartupMessages({
    library(rhdf5)
    library(SingleCellExperiment)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Pollen/pollen.rds")
mat <- as.matrix(normcounts(sce))  # No original counts available
cdata <- as.data.frame(colData(sce))

#clean ERCC
gene_mask <- !grepl("ERCC-", rownames(mat))
mat <- mat[gene_mask, ]

construct_dataset("../data/Pollen", mat, cdata)
message("Done!")
