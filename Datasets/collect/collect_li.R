#! /usr/bin/env Rscript
# by weil
# 4 Oct 2018
# 3:09 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Li/li.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))
meta_df$cell_type2 <- sapply(strsplit(rownames(meta_df), "_"), function(x){x[5]})
meta_df <- meta_df[, c("cell_type1", "cell_type2")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Li", expr_mat, meta_df, datasets_meta)
message("Done!")

