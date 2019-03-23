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
sce <- readRDS("../download/Hemberg/Manno_human/manno_human.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))

# Clean cell type
mask <- cdata$cell_type1 != "Unk"
count_mat <- count_mat[, mask]
cdata <- cdata[mask, c("cell_type1", "Source", "age", "WellID", "batch")]

construct_dataset("../data/Manno_human", count_mat, cdata)
message("Done!")
