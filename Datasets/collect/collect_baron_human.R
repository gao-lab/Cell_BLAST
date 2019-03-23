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
sce <- readRDS("../download/Hemberg/Baron_human/baron-human.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))
tmp <- sapply(strsplit(rownames(cdata), "\\."), function(x) x[1])
cdata$donor <- sapply(strsplit(tmp, "_"), function(x) x[1])
cdata$lib <- sapply(strsplit(tmp, "_"), function(x) x[2])
cdata <- cdata[, c("cell_type1", "donor", "lib")]
colnames(cdata)[3] <- "library"

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

construct_dataset("../data/Baron_human", count_mat, cdata, datasets_meta, cell_ontology, grouping = "donor")
message("Done!")
