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
sce <- readRDS("../download/Hemberg/Goolam/goolam.rds")
mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
cdata$source <- sapply(strsplit(rownames(cdata), "_"), function(x) {
    paste(x[-length(x)], collapse = "_")
})
cdata <- cdata[, c("cell_type1", "source")]

#clean ERCC
gene_mask <- !grepl("ERCC-", rownames(mat))
mat <- mat[gene_mask, ]

construct_dataset("../data/Goolam", mat, cdata)
message("Done!")
