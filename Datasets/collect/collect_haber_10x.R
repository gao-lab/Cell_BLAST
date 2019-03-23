#! /usr/bin/env Rscript
# by weil
# Aug 31, 2018
# 09:18 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Haber/10x/GSE92332_atlas_UMIcounts.txt")
meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df), "_"), function(x) x[3]))
meta_df$batch <- unlist(lapply(strsplit(rownames(meta_df), "_"), function(x) x[1]))
meta_df$region = "epithelial cells"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/small_intestine_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Haber_10x", as.matrix(expr_mat), meta_df, datasets_meta, 
                  cell_ontology, grouping = "batch")
message("Done!")



#FAE
expr_mat_FAE <- read.table("../download/Haber/10x/FAE_UMIcounts.txt.gz")
#without cell type annotation
meta_df_FAE <- data.frame(row.names = colnames(expr_mat_FAE))
meta_df_FAE$cellid = colnames(expr_mat_FAE)
meta_df_FAE$batch <- unlist(lapply(strsplit(rownames(meta_df_FAE), "_"), function(x) x[1]))
meta_df_FAE$cellid[meta_df_FAE$batch == "GFI1B."] <- paste0("GFI1B_", meta_df_FAE$cellid[meta_df_FAE$batch == "GFI1B."])
meta_df_FAE$cell_type1 <- unlist(lapply(strsplit(meta_df_FAE$cellid, "_"), function(x) x[4]))
meta_df_FAE$donor <- unlist(lapply(strsplit(meta_df_FAE$cellid, "_"), function(x) x[2]))
meta_df_FAE$batch <- unlist(lapply(strsplit(meta_df_FAE$cellid, "_"), function(x) x[1]))
meta_df_FAE$cellid <- NULL
meta_df_FAE$region = "follicle associated epithelia"

construct_dataset("../data/Haber_10x_FAE", as.matrix(expr_mat_FAE), meta_df_FAE, 
                  datasets_meta, cell_ontology, grouping = "batch")
message("Done!")



#spatial region
expr_mat_region <- read.table("../download/Haber/10x/GSE92332_Regional_UMIcounts.txt")
meta_df_region <- data.frame(row.names = colnames(expr_mat_region))
meta_df_region$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df_region), "_"), function(x) x[4]))
meta_df_region$region <- unlist(lapply(strsplit(rownames(meta_df_region), "_"), function(x) x[2]))
meta_df_region$donor <- unlist(lapply(strsplit(rownames(meta_df_region), "_"), function(x) x[3]))

construct_dataset("../data/Haber_10x_region", as.matrix(expr_mat_region), meta_df_region, 
                  datasets_meta, cell_ontology, grouping = "donor")
message("Done!")



#large cell/paneth enriched
expr_mat_largecell <- read.table("../download/Haber/10x/GSE92332_LargeCellSort_UMIcounts.txt")
meta_df_largecell <- data.frame(row.names = colnames(expr_mat_largecell))
meta_df_largecell$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df_largecell), "_"), function(x) x[4]))
meta_df_largecell$donor <- unlist(lapply(strsplit(rownames(meta_df_largecell), "_"), function(x) x[2]))
meta_df_largecell$batch <- unlist(lapply(strsplit(rownames(meta_df_largecell), "_"), function(x) x[3]))

construct_dataset("../data/Haber_10x_largecell", as.matrix(expr_mat_largecell), 
                  meta_df_largecell, datasets_meta, cell_ontology, grouping = "batch")
message("Done!")



###organoid
expr_mat_organoid <- read.table("../download/Haber/10x/GSE92332_Org_RANKL_UMIcounts.txt.gz")
meta_df_organoid <- data.frame(row.names = colnames(expr_mat_organoid))
meta_df_organoid$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df_organoid), "_"), function(x) x[4]))
meta_df_organoid$state <- unlist(lapply(strsplit(rownames(meta_df_organoid), "_"), function(x) x[3]))
meta_df_organoid$batch <- unlist(lapply(strsplit(rownames(meta_df_organoid), "_"), function(x) x[1]))
meta_df_organoid$organism = "Mus musculus"
meta_df_organoid$organ = "Small intestine organoid"
meta_df_organoid$platform = "10x"
meta_df_organoid$dataset_name = "Haber_10x_organoid"

construct_dataset("../data/Haber_10x_organoid", as.matrix(expr_mat_organoid), 
                  meta_df_organoid, datasets_meta, cell_ontology)
message("Done!")
