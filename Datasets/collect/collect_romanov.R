#! /usr/bin/env Rscript
# by weil
# 5 Oct 2018
# 5:24 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Romanov/romanov.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))
meta_df <- meta_df[, 1:11]
colnames(meta_df)[9] <- "gender"

construct_dataset("../data/Romanov", expr_mat, meta_df)
message("Done!")

