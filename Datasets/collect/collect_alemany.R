#! /usr/bin/env Rscript
# by weil
# Jan 19, 2019
# 02:13 PM

suppressPackageStartupMessages({
  library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
#kidney marrow 
expr_mat_wkm <- read.table("../download/Alemany/GSE102990_R1_R2_R4_wkm_transcriptome.txt.gz", 
                       header = TRUE, row.names = 1)
colnames(expr_mat_wkm) <- gsub("\\.", "-", colnames(expr_mat_wkm))
colnames(expr_mat_wkm) <- substring(colnames(expr_mat_wkm), 2, 15)

#meta_df_wkm
meta_df_wkm1 <- read.table("../download/Alemany/GSE102990_R1_R2_wkm_tsne.txt.gz",
                           row.names = 1, header = TRUE, stringsAsFactors = FALSE)
meta_df_wkm2 <- read.table("../download/Alemany/GSE102990_R4_wkm_tsne.txt.gz",
                           header = TRUE, stringsAsFactors = FALSE)
meta_df_wkm1 <- meta_df_wkm1 %>%
  select(
    ClusterID,
    cell_type1 = cellType
  )
meta_df_wkm2 <- meta_df_wkm2 %>%
  filter(
    CellType != "other"
  ) %>%
  select(
    CellID,
    ClusterID,
    cell_type1 = CellType
  )
rownames(meta_df_wkm2) <- meta_df_wkm2$CellID
meta_df_wkm2$CellID <- NULL
meta_df_wkm <- rbind(meta_df_wkm1, meta_df_wkm2)
meta_df_wkm$plate <- unlist(lapply(strsplit(rownames(meta_df_wkm), "-"), function(x) x[2]))
meta_df_wkm$donor <- unlist(lapply(strsplit(rownames(meta_df_wkm), "-"), function(x) x[4]))
meta_df_wkm$region = "marrow"
meta_df_wkm$lifestage = "adult"
meta_df_wkm$cell_type1 <- as.character(meta_df_wkm$cell_type1)

expr_mat_wkm <- expr_mat_wkm[, rownames(meta_df_wkm)]

#gene_meta_wkm
gene_meta_wkm <- data.frame(
  gene_id_name = rownames(expr_mat_wkm),
  Gene.stable.ID = unlist(lapply(strsplit(rownames(expr_mat_wkm), "_"), function(x) x[1])),
  row_gene_name = unlist(lapply(strsplit(rownames(expr_mat_wkm), "_"), function(x) x[2])),
  stringsAsFactors = FALSE)
zebrafish_94 <- read.csv("../../../GENOME/mapping/zebrafish_94.txt", stringsAsFactors = FALSE)
gene_meta_wkm <- merge(gene_meta_wkm, zebrafish_94, by = "Gene.stable.ID")
gene_meta_wkm <- gene_meta_wkm[!duplicated(gene_meta_wkm$Gene.name), ]
gene_meta_wkm <- gene_meta_wkm[gene_meta_wkm$Gene.name != "", ]
rownames(gene_meta_wkm) <- gene_meta_wkm$Gene.name
gene_meta_wkm$Gene.name <- NULL

expr_mat_wkm <- expr_mat_wkm[gene_meta_wkm$gene_id_name, ]
rownames(expr_mat_wkm) <- rownames(gene_meta_wkm)

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/zebrafish_kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Alemany_Kidney", as.matrix(expr_mat_wkm), meta_df_wkm, 
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology, gene_meta = gene_meta_wkm)
message("Done!")



#fin
expr_mat_fin <- read.table("../download/Alemany/GSE102990_R4_R5_fin_transcriptome.txt.gz",
                           sep = "\t", row.names = 1, header = TRUE)
colnames(expr_mat_fin) <- gsub("\\.", "-", colnames(expr_mat_fin))
colnames(expr_mat_fin) <- substring(colnames(expr_mat_fin), 2, 21)

#meta_df_fin
meta_df_fin <- read.table("../download/Alemany/GSE102990_R4_R5_fin_tsne.txt.gz",
                          sep = "\t", header = TRUE)
meta_df_fin <- meta_df_fin %>%
  filter(
    CellID != "outlier"
  ) %>%
  select(
    cell = X, 
    ClusterID,
    cell_type1 = CellID
  )
rownames(meta_df_fin) <- meta_df_fin$cell
meta_df_fin$cell <- NULL
meta_df_fin$cell_type1 <- as.character(meta_df_fin$cell_type1)
meta_df_fin$plate <- unlist(lapply(strsplit(rownames(meta_df_fin), "-"), function(x) x[2]))
meta_df_fin$donor <- unlist(lapply(strsplit(rownames(meta_df_fin), "-"), function(x) x[4]))
meta_df_fin$free_annotation <- unlist(lapply(strsplit(rownames(meta_df_fin), "-"), function(x) x[3]))
meta_df_fin$lifestage = "adult"

cells_fin <- intersect(rownames(meta_df_fin), colnames(expr_mat_fin))
meta_df_fin <- meta_df_fin[cells_fin, ]
expr_mat_fin <- expr_mat_fin[, rownames(meta_df_fin)]

#gene_meta_fin
gene_meta_fin <- data.frame(
  gene_id_name = rownames(expr_mat_fin),
  Gene.stable.ID = unlist(lapply(strsplit(rownames(expr_mat_fin), "_"), function(x) x[1])),
  row_gene_name = unlist(lapply(strsplit(rownames(expr_mat_fin), "_"), function(x) x[2])),
  stringsAsFactors = FALSE)
zebrafish_94 <- read.csv("../../../GENOME/mapping/zebrafish_94.txt", stringsAsFactors = FALSE)
gene_meta_fin <- merge(gene_meta_fin, zebrafish_94, by = "Gene.stable.ID")
gene_meta_fin <- gene_meta_fin[!duplicated(gene_meta_fin$Gene.name), ]
gene_meta_fin <- gene_meta_fin[gene_meta_fin$Gene.name != "", ]
rownames(gene_meta_fin) <- gene_meta_fin$Gene.name
gene_meta_fin$Gene.name <- NULL

expr_mat_fin <- expr_mat_fin[gene_meta_fin$gene_id_name, ]
rownames(expr_mat_fin) <- rownames(gene_meta_fin)

#assign cell ontology
cell_ontology_fin <- read.csv("../cell_ontology/zebrafish_fin_cell_ontology.csv")
cell_ontology_fin <- cell_ontology_fin[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

construct_dataset("../data/Alemany_Fin", as.matrix(expr_mat_fin), meta_df_fin, datasets_meta, cell_ontology_fin, grouping = "donor")
message("Done!")
