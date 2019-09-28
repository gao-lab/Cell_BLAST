#! /usr/bin/env Rscript
# by wangshuai
# 12 Apr 2019
# 18:19:35 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#READ metadata
cat("Reading metadata...\n")
cell_df <- read.table('../download/Green/GSE112393_MergedAdultMouseST25_PerCellAttributes.txt',
                      header = T,stringsAsFactors = F)
cell_df <- cell_df[,c(1,2,3)]
colnames(cell_df) <- c('cellID','donor','clusterID')
row.names(cell_df) <- cell_df$cellID
cell_df$lifestage <- 'adult'

#READ DGE
cat("Reading DGE\n")
exprs <- read.table('../download/Green/GSE112393_MergedAdultMouseST25_DGE.txt',
                    header = T,row.names = 1,stringsAsFactors = F)
inclued_cell <- intersect(row.names(cell_df),colnames(exprs))
exprs <- exprs[,inclued_cell]
cell_df <- cell_df[inclued_cell,]
exprs <- Matrix(as.matrix(exprs),sparse = T)
cell_ontology <- read.csv("../cell_ontology/mouse_spermatogenesis_ontology.csv")
cell_type  <- cell_ontology[,c("cell_type1","clusterID")]
cell_df <- merge(cell_df,cell_type,by="clusterID")
row.names(cell_df)<-cell_df$cellID
cell_df$cellID <- NULL
cell_df$clusterID <- NULL
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]


#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names=1)
construct_dataset("../data/Green", exprs, cell_df, datasets_meta, cell_ontology, grouping="donor", y_low = 0.1)
message("Done!")
