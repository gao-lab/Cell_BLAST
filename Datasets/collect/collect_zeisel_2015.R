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
sce <- readRDS("../download/Hemberg/Zeisel/zeisel.rds")
mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
cdata <- cdata[, c("clust_id", "cell_type1")]

construct_dataset("../data/Zeisel_2015", mat, cdata)
message("Done!")
