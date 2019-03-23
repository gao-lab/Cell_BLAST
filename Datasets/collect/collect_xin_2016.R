#! /usr/bin/env Rscript
# by caozj
# 18 Dec 2017
# 4:11:20 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods


suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(Seurat)
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Xin/xin.rds")
count_mat <- as.matrix(normcounts(sce))
cdata <- as.data.frame(colData(sce))

message("Filtering cells...")

# Clean cell type
mask <- !grepl("contaminated", cdata$cell_type1)
normal_mask <- cdata$condition == "Healthy"
disease_mask <- !normal_mask
cdata <- cdata %>% dplyr::select(
    donor=donor.id, disease = condition, lifestage= age, 
    ethnicity, gender, cell_type1
)
cdata$cell_id <- rownames(cdata)
cdata$lifestage <- paste0(cdata$lifestage, "-year")

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")

normal_count_mat <- count_mat[, mask & normal_mask]
normal_cdata <- cdata[mask & normal_mask, ] %>% dplyr::select(-disease)
rownames(normal_cdata) <- normal_cdata$cell_id
normal_cdata$cell_id <- NULL

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Xin_2016", normal_count_mat, normal_cdata, 
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology)

disease_count_mat <- count_mat[, mask & disease_mask]
disease_cdata <- cdata[mask & disease_mask, ] %>% dplyr::mutate(dataset_name = paste("Xin_2016", "disease", sep = "_"))
rownames(disease_cdata) <- disease_cdata$cell_id
disease_cdata$cell_id <- NULL
construct_dataset("../data/Xin_2016_disease", disease_count_mat, disease_cdata, 
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology)

message("Done!")
