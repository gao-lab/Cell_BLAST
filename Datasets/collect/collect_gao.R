#! /usr/bin/env Rscript
# by weil
# Nov 24, 2018
# 02:24 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.delim("../download/Gao/GSE95630_Digestion_TPM_new.txt")
rownames(expr_mat) <- expr_mat[,1]
expr_mat[,1] <- NULL
expr_mat <- as.matrix(expr_mat)

meta <- list()
annotation <- list()
metadata <- list()
for (i in 1:4){
  meta[[i]] <- read.xlsx("../download/Gao/organ_cluster.xlsx", sheet = i)
  annotation[[i]] <- read.xlsx("../download/Gao/organ_cluster_annotation.xlsx", sheet = i)
  metadata[[i]] <- merge(meta[[i]], annotation[[i]], by = "Cluster")
  if(i == 3) {metadata[[i]]$Tissue <- "small intestine"}
  metadata[[i]] <- metadata[[i]][, c("Sample", "Embryo", "Tissue", "Cluster", "cell_type1")]
}
meta_df <- rbind(metadata[[1]], metadata[[2]], metadata[[3]], metadata[[4]])
rownames(meta_df) <- meta_df[,1]
meta_df[,1] <- NULL
colnames(meta_df) <- c("donor", "organ", "cluster", "cell_type1")#cluster5 need to be identified
meta_df$organ <- gsub("esophagus", "Esophagus", meta_df$organ)
meta_df$organ <- gsub("stomach", "Stomach", meta_df$organ)
meta_df$organ <- gsub("small intestine", "Small Intestine", meta_df$organ)
meta_df$organ <- gsub("L-Intes", "Large Intestine", meta_df$organ)
meta_df$lifestage <- "fetal"

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

#Esophagus
meta_df_esophagus <- meta_df[meta_df$organ == "Esophagus", ]
expr_mat_esophagus <- expr_mat[,rownames(meta_df_esophagus)]
#assign cell ontology
cell_ontology_esophagus <- read.csv("../cell_ontology/esophagus_cell_ontology.csv")
cell_ontology_esophagus <- cell_ontology_esophagus[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]
construct_dataset("../data/Gao_Esophagus", expr_mat_esophagus, meta_df_esophagus,
                  datasets_meta, cell_ontology_esophagus)
message("Done!")

#Stomach
meta_df_stomach <- meta_df[meta_df$organ == "Stomach", ]
expr_mat_stomach <- expr_mat[,rownames(meta_df_stomach)]
#assign cell ontology
cell_ontology_stomach <- read.csv("../cell_ontology/stomach_cell_ontology.csv")
cell_ontology_stomach <- cell_ontology_stomach[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]
construct_dataset("../data/Gao_Stomach", expr_mat_stomach, meta_df_stomach,
                  datasets_meta, cell_ontology_stomach)
message("Done!")

#Small_intestine
meta_df_small_intestine <- meta_df[meta_df$organ == "Small Intestine", ]
expr_mat_small_intestine <- expr_mat[,rownames(meta_df_small_intestine)]
#assign cell ontology
cell_ontology_small_intestine <- read.csv("../cell_ontology/small_intestine_cell_ontology.csv")
cell_ontology_small_intestine <- cell_ontology_small_intestine[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]
construct_dataset("../data/Gao_Small_Intestine", expr_mat_small_intestine, meta_df_small_intestine,
                  datasets_meta, cell_ontology_small_intestine)
message("Done!")

#Large_intestine
meta_df_large_intestine <- meta_df[meta_df$organ == "Large Intestine", ]
expr_mat_large_intestine <- expr_mat[,rownames(meta_df_large_intestine)]
#assign cell ontology
cell_ontology_large_intestine <- read.csv("../cell_ontology/large_intestine_cell_ontology.csv")
cell_ontology_large_intestine <- cell_ontology_large_intestine[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]
construct_dataset("../data/Gao_Large_Intestine", expr_mat_large_intestine, meta_df_large_intestine,
                  datasets_meta, cell_ontology_large_intestine)
message("Done!")
