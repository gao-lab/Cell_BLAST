#! /usr/bin/env Rscript
# by weil
# 28 Sep 2018
# 5:00 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Manno_mouse/manno_mouse.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))

#clean cell type
mask <- meta_df$cell_type1 != "mUnk"
expr_mat <- expr_mat[, mask]
meta_df <- meta_df[mask, c("Species", "cell_type1", "Source", "age", "WellID", "batch")]

construct_dataset("../data/Manno_mouse", expr_mat, meta_df)
message("Done!")

