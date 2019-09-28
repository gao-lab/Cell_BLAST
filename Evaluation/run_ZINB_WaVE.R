#! /usr/bin/env Rscript
# by caozj
# 9 Feb 2018
# 7:52:31 PM


source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(zinbwave)
    library(rhdf5)
})
source("../Utilities/data.R")

parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", default = NULL)
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 2)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
parser$add_argument("-j", "--n-jobs", dest = "n_jobs", type = "integer", default = 1)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()

BiocParallel::register(BiocParallel::MulticoreParam(args$n_jobs))


# Read data
cat("[Info] Reading data...\n")
dataset <- read_dataset(args$input)
if (!is.null(args$clean))
    dataset <- clean_dataset(dataset, args$clean)
if (!is.null(args$genes))
    dataset <- dataset[dataset@uns[[args$genes]], ]
x <- round(dataset@exprs)


# Run ZINB-WaVE
if (!is.null(args$seed)) {
    set.seed(args$seed)
}
cat("[Info] Running ZINB-WaVE\n")
start_time <- proc.time()
result <- zinbFit(as.matrix(x), K = args$dim, maxiter.optimize = 1e4)  # Allow convergence
elapsed_time <- proc.time() - start_time


# Save results
cat("[Info] Saving results...\n")
write_hybrid_path(t(getW(result)), paste(args$output, "latent", sep = "//"))
write_hybrid_path(elapsed_time["elapsed"],
                  paste(args$output, "time", sep = "//"))

cat("[Info] Done!\n")
