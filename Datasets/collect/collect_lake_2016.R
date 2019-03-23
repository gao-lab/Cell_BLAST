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
sce <- readRDS("../download/Hemberg/Lake/lake.rds")
expr_mat <- as.matrix(normcounts(sce))
cdata <- as.data.frame(colData(sce))

meta_df <- cdata[, c("cell_type1", "Source", "age", "WellID", "batch", "Plate")]
colnames(meta_df) <- c(
    "cell_type1", "tissue", "age", "well", "batch", "plate"
)

construct_dataset("../data/Lake_2016", expr_mat, meta_df)
message("Done!")
