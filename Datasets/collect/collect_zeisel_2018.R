#! /usr/bin/env Rscript
# by weil
# Sep 04, 2018
# 01:04 PM

suppressPackageStartupMessages({
    library(loomR)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading expression data...")
expr <- connect(filename = "../download/Zeisel_2018/l5_all.loom", mode = "r+", skip.validate = TRUE)
expr_mat <- t(expr[["matrix"]][,])
genenames <- expr[["row_attrs/Gene"]][]
accession <- expr[["row_attrs/Accession"]][]
rownames(expr_mat) <- accession
colnames(expr_mat) <- as.character(1:dim(expr_mat)[2])

message("Reading metadata...")
cellid <- expr[["col_attrs/CellID"]][]
age <- expr[["col_attrs/Age"]][]
donor <- expr[["col_attrs/DonorID"]][]
tissue <- expr[["col_attrs/Region"]][]
strain <- expr[["col_attrs/Strain"]][]
taxonomy <- expr[["col_attrs/TaxonomyRank4"]][]

meta_df <- data.frame(
    row.names = colnames(expr_mat),
    cellid = cellid,
    cell_type1 = taxonomy,
    age = age,
    donor = donor,
    region = tissue,
    strain = strain
)

meta_gene = data.frame(row.names = rownames(expr_mat), genenames = genenames)

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/nervous_system_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Zeisel_2018", expr_mat, meta_df, datasets_meta, cell_ontology, gene_meta = meta_gene)
message("Done!")
