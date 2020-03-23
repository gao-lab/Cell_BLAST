#! /usr/bin/env Rscript
# by caozj
# Jan 13, 2020
# 2:37:41 PM

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(Rtsne)
})
source("../Utilities/data.R")

parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 2)
parser$add_argument("-j", "--n-jobs", dest = "n_jobs", type = "integer", default = 1)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
args <- parser$parse_args()

cat("[Info] Reading and preprocessing...\n")
x <- t(read_hybrid_path(args$input))

if (!is.null(args$seed)) {
    set.seed(args$seed)
}

cat("[Info] Running tSNE...\n")
start_time <- proc.time()
tsne.rs <- Rtsne(x, dims = args$dim, pca = FALSE, num_threads = args$n_jobs)
elapsed_time <- proc.time() - start_time

cat("[Info] Saving results...\n")
write_hybrid_path(t(tsne.rs$Y), paste(args$output, "latent", sep = "//"))
time_path <- paste(args$output, "time", sep = "//")
if (check_hybrid_path(time_path))
    elapsed_time["elapsed"] <-
        elapsed_time["elapsed"] + read_hybrid_path(time_path)
write_hybrid_path(elapsed_time["elapsed"], time_path)

cat("[Info] Done!\n")
