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
sce <- readRDS("../download/Hemberg/Macosko/macosko.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
cdata <- cdata[, c("cell_type1", "clust_id")]
colnames(cdata)[2] <- "cluster"

cell_ontology <- read.csv("../cell_ontology/retina_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Macosko", count_mat, cdata, datasets_meta, cell_ontology)
message("Done!")
