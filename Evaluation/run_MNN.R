#! /usr/bin/env Rscript
# by caozj
# Jun 2, 2018
# 4:50:42 PM

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(Matrix)
    library(scran)
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

# Read data
cat("Reading and preprocessing...\n")
dataset <- read_dataset(args$input)
dataset <- normalize(dataset)
if (!is.null(args$clean))
    dataset <- clean_dataset(dataset, args$clean)
if (!is.null(args$genes))
    dataset <- dataset[dataset@uns[[args$genes]], ]
start_time <- proc.time()
dataset@exprs <- log1p(dataset@exprs)

arg_list <- list()
for (batch in unique(dataset@obs[[args$batch]])) {
    mask <- dataset@obs[[args$batch]] == batch
    arg_list[[batch]] <- as.matrix(dataset@exprs[, mask])
}
names(arg_list) <- NULL
arg_list[["cos.norm.out"]] <- FALSE
arg_list[["BPPARAM"]] <- MulticoreParam(args$n_jobs)

# MNN
cat("Performing MNN correction...\n")
mnn_result <- do.call(mnnCorrect, arg_list)

# Save result
cat("Saving result...\n")
new_exprs <- as.matrix(Reduce(cbind, mnn_result$corrected))
elapsed_time <- proc.time() - start_time
write_hybrid_path(new_exprs, paste(args$output, "exprs", sep = "//"))
write_hybrid_path(elapsed_time["elapsed"], paste(args$output, "time", sep = "//"))

message("Done!")
