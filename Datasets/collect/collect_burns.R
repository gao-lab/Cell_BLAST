#! /usr/bin/env Rscript
# by wangshuai
# 29 Apr 2019
# 21:12:52 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#READ metadata
cat("Reading metadata...\n")
cell_df1 = read.table('../download/Burns/GSE71982_P1_Coch__FACs__PhenoData.txt',
                       header = T,stringsAsFactors = F)
cell_df1 = cell_df1[,1:2]
colnames(cell_df1) <- c('cell_names','cell_type1')
cell_df2 = read.csv('../download/Burns/GSE71982_P1_Coch__nonFACs__PhenoData.csv',
                       header = T,stringsAsFactors = F)
cell_df2 = cell_df2[,1:2]
colnames(cell_df2) <- c('cell_names','cell_type1')
cell_df3 = read.csv('../download/Burns/GSE71982_P1_Utricle_PhenoData.csv',
                       header = T,stringsAsFactors = F)
cell_df3 = cell_df3[,1:2]
colnames(cell_df3) <- c('cell_names','cell_type1')
cell_df = rbind(cell_df1,cell_df2,cell_df3)   
          

colnames(cell_df) <- c('cell_names','cell_type1')
cell_df$lifestage <- 'newborn'
row.names(cell_df) <- cell_df[,1]
cell_df$cell_names <- NULL

# Reading DGE
cat("Reading DGE\n")
exprs <- read.csv('../download/Burns/GSE71982_RSEM_Counts_Matrix.csv',
                    header = T,row.names=1,stringsAsFactors = F)
inclued_cell <- intersect(row.names(cell_df),colnames(exprs))
exprs <- exprs[,inclued_cell]
cell_df <- cell_df[inclued_cell,]
exprs <- Matrix(as.matrix(exprs),sparse = T)
genes <- rownames(exprs)
genes <- gsub('"','',genes)
rownames(exprs) <- genes

# Reading ontology
cell_ontology <- read.csv("../cell_ontology/mouse_inner_ear_cell_ontology.csv",
                          header = T,stringsAsFactors = F)
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

# Constructing dataset
cat("Constructing dataset\n")
datasets_meta <- read.csv("../ACA_datasets_2.csv", header = TRUE, row.names=1)
construct_dataset("../data/Burns", exprs, cell_df, datasets_meta, cell_ontology, y_low = 0.1)
message("Done!")
