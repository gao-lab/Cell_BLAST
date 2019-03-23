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
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Segerstolpe/segerstolpe.rds")
count_mat <- as.matrix(counts(sce))
cdata <- as.data.frame(colData(sce))

message("Filtering...")

# Clean up cell type
mask <- cdata$cell_quality == "OK" & ! cdata$cell_type1 %in% c(
    "not applicable", "co-expression", "unclassified")
normal_mask <- cdata$disease == "normal"
disease_mask <- !normal_mask
cdata <- cdata %>% dplyr::select(cell_type1, lifestage=age, disease, gender=sex)
cdata$lifestage <- paste0(cdata$lifestage, "-year")
cdata$cell_id <- rownames(cdata)

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")

# Clean up ERCC
ercc_mask <- grepl("^ERCC_", rownames(count_mat))
count_mat <- count_mat[!ercc_mask, ]

normal_count_mat <- count_mat[, mask & normal_mask]
normal_cdata <- cdata[mask & normal_mask, ] %>% dplyr::select(-disease)
rownames(normal_cdata) <- normal_cdata$cell_id
normal_cdata$cell_id <- NULL
#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Segerstolpe", normal_count_mat, normal_cdata, datasets_meta, cell_ontology)

### disease
disease_count_mat <- count_mat[, mask & disease_mask]
disease_cdata <- cdata[mask & disease_mask, ] %>%
  dplyr::mutate(dataset_name = paste("Segerstolpe", "disease", sep="_"))
rownames(disease_cdata) <- disease_cdata$cell_id
disease_cdata$cell_id <- NULL

construct_dataset("../data/Segerstolpe_disease", disease_count_mat, disease_cdata, datasets_meta, cell_ontology)
message("Done!")
