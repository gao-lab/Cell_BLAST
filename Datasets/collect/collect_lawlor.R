#! /usr/bin/env Rscript
# by caozj
# May 23, 2018
# 7:09:35 PM

suppressPackageStartupMessages({
    library(GEOquery)
    library(Seurat)
    library(biomaRt)
    library(Hmisc)
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
data_path <- "../download/Lawlor"
es <- getGEO(filename = file.path(data_path, "GSE86469_series_matrix.txt.gz"))
meta_df <- pData(es) %>% dplyr::select(
    cell_id = `title`,
    lifestage = `age:ch1`,
    bmi = `bmi:ch1`,
    cell_type1 = `cell type:ch1`,
    disease = `disease:ch1`,
    ethnicity = `race:ch1`,
    gender = `Sex:ch1`
)
meta_df$lifestage <- paste0(meta_df$lifestage, "-year")
expr_mat <- as.matrix(read.csv(file.path(
    data_path, "GSE86469_GEO.islet.single.cell.processed.data.RSEM.raw.expected.counts.csv.gz"
), row.names = 1, check.names = FALSE))
stopifnot(all(colnames(expr_mat) == meta_df$cell_id))

mapping <- read.table("../../../GENOME/mapping/dec2013_ensg_hgnc_clean.txt",
                      stringsAsFactors = FALSE, header = TRUE)
expr_mat <- expr_mat[mapping$ensembl_gene_id, ]
rownames(expr_mat) <- mapping$hgnc_symbol

message("Filtering...")
mask <- meta_df$cell_type1 != "None/Other"
normal_mask <- meta_df$disease == "Non-Diabetic"
disease_mask <- !normal_mask

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")
cell_ontology$cell_type1 <- capitalize(as.character(cell_ontology$cell_type1))

normal_expr_mat <- expr_mat[, mask & normal_mask]
normal_meta_df <- meta_df[mask & normal_mask, ] %>% dplyr::select(-disease)
rownames(normal_meta_df) <- normal_meta_df$cell_id
normal_meta_df$cell_id <- NULL

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Lawlor", normal_expr_mat, normal_meta_df, datasets_meta, cell_ontology)

disease_expr_mat <- expr_mat[, mask & disease_mask]
disease_meta_df <- meta_df[mask & disease_mask, ] %>% dplyr::mutate(dataset_name = paste("Lawlor", "disease", sep = "_"))
rownames(disease_meta_df) <- disease_meta_df$cell_id
disease_meta_df$cell_id <- NULL
construct_dataset("../data/Lawlor_disease", disease_expr_mat, disease_meta_df, datasets_meta, cell_ontology)

message("Done!")
