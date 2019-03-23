#! /usr/bin/env Rscript
# by caozj
# Jun 28, 2018
# 11:51:49 AM

suppressPackageStartupMessages({
    library(argparse)
    library(SingleCellExperiment)
    library(scmap)
})
source("../../Utilities/data.R", chdir = TRUE)

parser <- ArgumentParser()
parser$add_argument("-d", "--data", dest = "data",
                    type = "character", required = TRUE)
parser$add_argument("-u", "--uns-slot", dest = "uns_slot",
                    type = "character", default = "scmap_genes")
parser$add_argument("-n", "--n-features", dest = "n_features",
                    type = "integer", default = 500)
args <- parser$parse_args()

dataset <- read_dataset(args$data)
dataset <- normalize(dataset)
sce <- SingleCellExperiment(
    assays = list(normcounts = as.matrix(dataset@exprs)),
    colData = dataset@obs
)
logcounts(sce) <- log1p(normcounts(sce))
rowData(sce)$feature_symbol <- rownames(sce)
sce <- selectFeatures(sce, n_features = args$n_features, suppress_plot = TRUE)
selected_genes <- sort(rownames(sce)[rowData(sce)$scmap_features])

write_hybrid_path(selected_genes, sprintf(
    "%s//uns/%s", args$data, args$uns_slot))
