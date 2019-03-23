#! /usr/bin/env Rscript
# by weil
# Dec 16, 2018
# 04:17 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- as.matrix(read.table(
  "../download/Adam/GSM2473317_run1855_out_gene_exon_tagged.dge.txt.gz",
  sep = "\t", header = TRUE, row.names = 1
))

#metadata
meta_df_CM <- read.delim("../download/Adam/GSE94333_P1_Cold_Cap_Mesenchyme.txt.gz", header = F)
meta_df_CM$cell_type1 <- "CM"
meta_df_DT <- read.delim("../download/Adam/GSE94333_P1_Cold_Distal_Tubule.txt.gz", header = F)
meta_df_DT$cell_type1 <- "Distal tubule"
meta_df_ED <- read.delim("../download/Adam/GSE94333_P1_Cold_Endothelial.txt.gz", header = F)
meta_df_ED$cell_type1 <- "ED"
meta_df_LH <- read.delim("../download/Adam/GSE94333_P1_Cold_Loop_of_Henle.txt.gz", header = F)
meta_df_LH$cell_type1 <- "LH"
meta_df_podo <- read.delim("../download/Adam/GSE94333_P1_Cold_Podocytes.txt.gz", header = F)
meta_df_podo$cell_type1 <- "Podocytes"
meta_df_PT <- read.delim("../download/Adam/GSE94333_P1_Cold_Proximal_Tubule.txt.gz", header = F)
meta_df_PT$cell_type1 <- "PT"
meta_df_stromal <- read.delim("../download/Adam/GSE94333_P1_Cold_Stromal.txt.gz", header = F)
meta_df_stromal$cell_type1 <- "Stromal"
meta_df_ub <- read.delim("../download/Adam/GSE94333_P1_Cold_Ureteric_Bud.txt.gz", header = F)
meta_df_ub$cell_type1 <- "Ureteric bud"
meta_df <- rbind(meta_df_CM, meta_df_DT, meta_df_ED, meta_df_LH, meta_df_podo, meta_df_PT, meta_df_stromal, meta_df_ub)

rownames(meta_df) <- meta_df$V1
meta_df$V1 <- NULL
meta_df$lifestage = "post natal day 1"

expr_mat <- expr_mat[, rownames(meta_df)]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

construct_dataset("../data/Adam", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
