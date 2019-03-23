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
sce <- readRDS("../download/Hemberg/Deng/deng-rpkms.rds")
count_mat <- as.matrix(normcounts(sce))
cdata <- as.data.frame(colData(sce))
cdata <- cdata[, c("cell_type1", "cell_type2")]

construct_dataset("../data/Deng", count_mat, cdata)
message("Done!")
