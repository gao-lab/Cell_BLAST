#! /usr/bin/env Rscript
# by weil
# Jan 18, 2019
# 09:23 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Karaiskos_drosophila/GSE95025_high_quality_cells_digital_expression.txt.gz", 
                        header = TRUE, row.names = 1)

#meta_df
meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cluster <- unlist(lapply(strsplit(rownames(meta_df), "_"), function(x) x[2]))
meta_df$lifestage = "embryo"

#assign cell ontology
#cell_ontology <- read.csv("../cell_ontology/")
#cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Karaiskos_drosophila", as.matrix(expr_mat), meta_df, datasets_meta)
message("Done!")
