#! /usr/bin/env Rscript
# by wangshuai
# 17 May 2019
# 18:47:52 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#READ metadata
cat("Reading metadata...\n")
cell_df = read.csv('../download/Meredith/GSE70798_metadata_WT.csv',
                      header = T,row.names = 1,stringsAsFactors = F)
cell_df <- cell_df[,c(1,3,6)]
colnames(cell_df) <- c('cell_type1','organ','life stage')
cell_df$age <- '5 weeks'

# Reading DGE
cat("Reading DGE\n")
exprs <- read.csv('../download/Meredith/GSE70798_SCS_MEC.csv.gz',
                  header = T,row.names=1,stringsAsFactors = F)
inclued_cell <- intersect(row.names(cell_df),colnames(exprs))
exprs <- exprs[,inclued_cell]
cell_df <- cell_df[inclued_cell,]
exprs <- Matrix(as.matrix(exprs),sparse = T)

# Reading ontology
cell_ontology <- read.csv("../cell_ontology/mouse_thymus_cell_ontology.csv",
                          header = T,stringsAsFactors = F)
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

# Constructing dataset
cat("Constructing dataset\n")
datasets_meta <- read.csv("../ACA_datasets_2.csv", header = TRUE, row.names=1)
construct_dataset("../data/Meredith", exprs, cell_df, datasets_meta, cell_ontology, y_low = 0.1)
message("Done!")