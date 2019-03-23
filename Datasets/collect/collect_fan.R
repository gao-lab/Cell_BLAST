#! /usr/bin/env Rscript
# by weil
# 4 Oct 2018
# 7:40 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Fan/fan.rds")
expr_mat <- as.matrix(normcounts(sce))
meta_df <- as.data.frame(colData(sce))

construct_dataset("../data/Fan", expr_mat, meta_df)
message("Done!")

