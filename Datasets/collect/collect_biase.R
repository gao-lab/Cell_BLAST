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
sce <- readRDS("../download/Hemberg/Biase/biase.rds")
mat <- as.matrix(normcounts(sce))  # No original counts available
cdata <- as.data.frame(colData(sce))

construct_dataset("../data/Biase", mat, cdata)
message("Done!")
