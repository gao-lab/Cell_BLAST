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
sce <- readRDS("../download/Hemberg/Yan/yan.rds")
mat <- as.matrix(normcounts(sce))  # No original counts available
cdata <- as.data.frame(colData(sce))
cdata$source <- sapply(strsplit(rownames(cdata), "\\.\\."), function(x) {
    paste(x[-length(x)], collapse = "..")
})
cdata$source[1:6] <- c(
    paste("Oocyte", 1:3, sep = ".."),
    paste("Zygote", 1:3, sep = "..")
)

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

construct_dataset("../data/Yan", mat, cdata, datasets_meta)
message("Done!")
