#! /usr/bin/env Rscript
# by caozj
# Jul 18, 2018
# 2:43:29 PM

# Strange behavior: cannot run in Rscript

suppressPackageStartupMessages({
    library(argparse)
    library(SCMarker)
})
source("../../Utilities/data.R", chdir = TRUE)

parser <- ArgumentParser()
parser$add_argument("-d", "--data", dest = "data", type = "character", required = TRUE)
parser$add_argument("-u", "--uns-slot", dest = "uns_slot", type = "character", default = "scmarker_genes")
args <- parser$parse_args()

# args <- list(
#     data = "../data/Worm/data_neural.h5",
#     uns_slot = "scmarker_genes"
# )

dataset <- read_dataset(args$data)
dataset <- normalize(dataset)
dataset@exprs <- log1p(dataset@exprs)

filterres <- ModalFilter(
    data = as.matrix(dataset@exprs),
    geneK = 0, cellK = 10, width = 2, cutoff = 2)
filterres <- GeneFilter(filterres = filterres, maxexp = ncol(filterres$data))
filterres <- getMarker(filterres = filterres, MNN = 200, MNNIndex = 20)

write_hybrid_path(filterres$marker, sprintf(
    "%s//uns/%s", args$data, args$uns_slot))
message("Done!")
