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
sce <- readRDS("../download/Hemberg/Shekhar/shekhar.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
cdata$replicate <- sapply(strsplit(rownames(cdata), "_"), function(x) x[1])
cdata$batch <- ifelse(cdata$replicate %in% c("Bipolar5", "Bipolar6"), "batch2", "batch1")

# Clean up cell type
mask <- cdata$cell_type1 != "unknown"
cdata <- cdata[mask, c("cell_type1", "cell_type2", "clust_id", "replicate", "batch")]
colnames(cdata)[3] <- "cluster"
count_mat <- count_mat[, mask]

cell_ontology <- read.csv("../../Datasets/cell_ontology/retina_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Shekhar", count_mat, cdata, datasets_meta, cell_ontology, grouping = "batch")
message("Done!")
