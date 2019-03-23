#! /usr/bin/env Rscript
# by weil
# Aug 15, 2018
# 07:23 PM

suppressPackageStartupMessages({
    library(loomR)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading expression data...")
expr <- connect(filename = "../download/Davie/Aerts_Fly_AdultBrain_Filtered_57k.loom", mode = "r+", skip.validate = TRUE)
expr_mat <- t(expr[["matrix"]][,])
genenames <- expr[["row_attrs/Gene"]][]
cellid <- expr[["col_attrs/CellID"]][]
rownames(expr_mat) <- genenames
colnames(expr_mat) <- cellid

message("Reading metadata...")
cluster <- expr[["col_attrs/Clusterings"]][]
cluster[,"cellid"] <- cellid
clusterid <- cluster[,c("0","cellid")]
colnames(clusterid) <- c("Cluster.ID", "cellid")

clusterannotation <- as.matrix(read.csv("../download/Davie/cluster_annotation.csv", header = TRUE))[1:87,1:2]
clustertype <- merge(clusterid, clusterannotation, by = "Cluster.ID")
rownames(clustertype) <- clustertype[, "cellid"]
clustertype <- clustertype[cellid, ]
clustertype[, "cellid"] <- NULL

age <- expr[["col_attrs/Age"]][]
gender <- expr[["col_attrs/Gender"]][]

meta_df <- data.frame(
	row.names = cellid,
  cell_type1 = clustertype[,"Annotation"],
	cluster = clustertype[,"Cluster.ID"],
	age = age,
	gender = gender
)

#clean cell type
mask <- meta_df$cell_type1 != "Unannotated"
meta_df <- meta_df[mask, ]
expr_mat <- expr_mat[, mask]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/drosophila_brain_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset(save_dir = "../data/Davie", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
