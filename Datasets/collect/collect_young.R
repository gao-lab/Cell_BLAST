#! /usr/bin/env Rscript
# by caozj
# Jan 1, 2018

suppressPackageStartupMessages({
    library(openxlsx)
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- readMM("../download/Young/tableOfCounts.mtx")
colLabels <- read.table("../download/Young/tableOfCounts_colLabels.tsv", header = TRUE)
rowLabels <- read.table("../download/Young/tableOfCounts_rowLabels.tsv", header = TRUE)
gene_mask <- !duplicated(rowLabels$Symbol)
expr_mat <- expr_mat[gene_mask, ]
rowLabels <- rowLabels[gene_mask, ]
rownames(expr_mat) <- rowLabels$Symbol
colnames(expr_mat) <- colLabels$DropletID

patient_info <- read.xlsx(
    "../download/Young/aat1699-Young-TablesS1-S12-revision1.xlsx",
    sheet = "TableS1 - Patient manifest", startRow = 2
)
cluster_info <- read.xlsx(
    "../download/Young/aat1699-Young-TablesS1-S12-revision1.xlsx",
    sheet = "TableS2 - Cluster info", startRow = 2
)
sample_info <- read.xlsx(
    "../download/Young/aat1699-Young-TablesS1-S12-revision1.xlsx",
    sheet = "TableS6 - Sample manifest", startRow = 2
)
cell_info <- read.table("../download/Young/cellManifestCompressed.tsv.gz", header = TRUE)

patient_info <- patient_info %>%
    select(donor = `Donor.(study.ID)`, Experiment, age = Age)
cluster_info <- cluster_info %>%
    select(
        ClusterID = Cluster_ID,
        cell_type0 = Cell_type1,
        cell_type1 = Cell_type2,
        cell_type2 = Cell_type3
    )
sample_info <- sample_info %>%
    select(
        Label, Experiment, organ = Organ, region = Location2,
        BiologicalRepNo, TechnicalRepNo, AgeInMonthsPostConception
    )
cell_info <- cell_info %>%
    filter(QCpass, Compartment == "Normal_Epithelium_and_Vascular_without_PT") %>%
    select(DropletID, ClusterID, Label = Source)

meta_df <- merge(sample_info, patient_info, by="Experiment")
meta_df <- merge(meta_df, cell_info, by="Label")
meta_df <- merge(meta_df, cluster_info, by="ClusterID")

meta_df <- meta_df %>%
    filter(
        cell_type1 != "-",
        cell_type1 != "Distal_tubules_and_collecting_duct"
    ) %>%
    select(-one_of("Label", "Experiment"))

rownames(meta_df) <- meta_df$DropletID
meta_df$DropletID <- NULL

expr_mat <- expr_mat[, rownames(meta_df)]

# Assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Young", expr_mat, meta_df, datasets_meta, cell_ontology, grouping = "donor", min_group_frac = 0.25)
message("Done!")
