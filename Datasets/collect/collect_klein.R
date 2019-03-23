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
sce <- readRDS("../download/Hemberg/Klein/klein.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
cdata$cell_type1 <- as.character(cdata$cell_type1)
cdata = cdata[, "cell_type1", drop = FALSE]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Klein", count_mat, cdata, datasets_meta)
message("Done!")
