#! /usr/bin/env Rscript
# by caozj
# Jun 17, 2018
# 3:12:45 PM

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(sva)
    library(argparse)
})
source("../Utilities/data.R", chdir = TRUE)

parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-b", "--batch", dest = "batch", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", default = NULL)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-j", "--n-jobs", dest = "n_jobs", type = "integer", default = 1)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()

# Reading and preprocessing
cat("Reading and preprocessing...\n")
dataset <- read_dataset(args$input)
data <- normalize(dataset)
if (!is.null(args$clean))
    dataset <- clean_dataset(dataset, args$clean)
if (!is.null(args$genes))
    dataset <- dataset[dataset@uns[[args$genes]], ]
start_time <- proc.time()
dataset@exprs <- log1p(dataset@exprs)

# Perform ComBat
cat("Performing ComBat...\n")
combat_expr_mat <- ComBat(
    dat = as.matrix(dataset@exprs),
    batch = dataset@obs[[args$batch]],
    mod = model.matrix(~1, data = dataset@obs),
    BPPARAM = MulticoreParam(args$n_jobs)
)

# Save result
cat("Saving result...\n")
elapased_time <- proc.time() - start_time
write_hybrid_path(combat_expr_mat, paste(args$output, "exprs", sep = "//"))
write_hybrid_path(elapased_time["elapsed"], paste(args$output, "time", sep = "//"))
cat("Done!\n")