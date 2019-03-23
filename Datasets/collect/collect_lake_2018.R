#! /usr/bin/env Rscript
# by weil
# Aug 13, 2018
# 01:16 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
data_path <- "../download/Lake_humanbrain"
expr_mat1 <- read.table(file.path(
    data_path, "GSE97930_CerebellarHem_snDrop-seq_UMI_Count_Matrix_08-01-2017.txt"
),  check.names = FALSE)
expr_mat2 <- read.table(file.path(
    data_path, "GSE97930_FrontalCortex_snDrop-seq_UMI_Count_Matrix_08-01-2017.txt"
),  check.names = FALSE)
expr_mat3 <- read.table(file.path(
    data_path, "GSE97930_VisualCortex_snDrop-seq_UMI_Count_Matrix_08-01-2017.txt"
),  check.names = FALSE)

message("merging three datasets...")
expr_mat1$gene <- rownames(expr_mat1)
expr_mat2$gene <- rownames(expr_mat2)
expr_mat3$gene <- rownames(expr_mat3)
expr_mat12 <- merge(expr_mat1,expr_mat2, by = "gene", all = TRUE)
expr_mat <- merge(expr_mat12,expr_mat3, by = "gene", all = TRUE)
rownames(expr_mat) <- expr_mat$gene
expr_mat$gene <- NULL
expr_mat <- as.matrix(expr_mat)
expr_mat[is.na(expr_mat)] <- 0

cells <- colnames(expr_mat)
cells_split <- strsplit(cells, "_")
cell_type1 <- NULL
tissue <- NULL
donor <- NULL
for(i in 1:length(cells)){
	cell_split <- cells_split[[i]]
	cell_type1[i] = cell_split[1]
	tissue[i] = substring(cell_split[2], 1, 3)
	donor[i] = substring(cell_split[2], 4, 5)
}

meta_df <- data.frame(
  row.names = colnames(expr_mat), 
  cell_type1 = cell_type1, 
  region = tissue, 
  donor = donor
)

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/human_brain_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Lake_2018", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
