#! /usr/bin/env Rscript
# by caozj
# 24 Jan 2018
# 11:14:15 PM


source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(Matrix)
    library(irlba)
    library(dplyr)
    library(rhdf5)
})
source("../Utilities/data.R")

eig.pca <- function(x, dim, scale = TRUE) {
    x <- scale(x, center = TRUE, scale = scale)
    cat("[Info] Using eigenvalue decomposition for computing PCA...\n")
    rotation <- eigen(t(x) %*% x, symmetric = TRUE)$vectors[, 1:dim]
    list(
        y = x %*% rotation,
        rotation = rotation
    )
}

svd.pca <- function(x, dim, scale = TRUE) {
    x <- scale(x, center = TRUE, scale = scale)
    cat("[Info] Using SVD for computing PCA...\n")
    svd.rs <- svd(x, nu = 0, nv = dim)
    list(
        y = svd.rs$u[, 1:dim] %*% diag(svd.rs$d[1:dim]),
        rotation = svd.rs$v
    )
}

irlba.pca <- function(x, dim, scale = TRUE) {
    x <- scale(x, center = TRUE, scale = scale)
    cat("[Info] Using irlba for computing PCA...\n")
    svd.rs <- irlba(x, nv = dim)
    list(
        y = svd.rs$u %*% diag(svd.rs$d),
        rotation = svd.rs$v
    )
}

adaptive.pca <- function(x, dim, scale = TRUE) {
    if (dim <= 0.5 * min(dim(x))) {
        irlba.pca(x, dim, scale)
    } else if (dim(x)[1] < dim(x)[2]) {
        svd.pca(x, dim, scale)
    } else {
        eig.pca(x, dim, scale)
    }
}

parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", default = NULL)
parser$add_argument("-l", "--log", dest = "log", default = FALSE, action = "store_true")
parser$add_argument("-c", "--scale", dest = "scale", default = FALSE, action = "store_true")
parser$add_argument("-m", "--method", dest = "method", type = "character",
                    choices = c("svd", "eig", "irlba", "adaptive"), default = "adaptive")
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 2)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()

# Read data
cat("[Info] Reading and preprocessing...\n")
if (grepl("//", args$input)) {  # Still support PCA on pure matrices
    x <- t(read_hybrid_path(args$input))
} else {
    dataset <- read_dataset(args$input)
    dataset <- normalize(dataset)
    if (!is.null(args$clean))
        dataset <- clean_dataset(dataset, args$clean)
    if (!is.null(args$genes))
        dataset <- dataset[dataset@uns[[args$genes]], ]
    x <- t(dataset@exprs)
}

if (args$method == "svd") {
    pca <- svd.pca
} else if (args$method == "eig") {
    pca <- eig.pca
} else if (args$method == "irlba") {
    pca <- irlba.pca
} else {  # args$method == "adaptive"
    pca <- adaptive.pca
}

# Run PCA
if (!is.null(args$seed)) {
    set.seed(args$seed)
}
cat("[Info] Running PCA...\n")
start_time <- proc.time()
if (args$log)
    x <- log1p(x)
pca.rs <- pca(x, dim = args$dim, scale = args$scale)
elapsed_time <- proc.time() - start_time

# Save results
cat("[Info] Saving results...\n")
write_hybrid_path(t(pca.rs$y), paste(args$output, "latent", sep = "//"))
write_hybrid_path(t(pca.rs$rotation), paste(args$output, "rotation", sep = "//"))
write_hybrid_path(dataset@uns[[args$genes]], paste(args$output, "genes", sep = "//"))
write_hybrid_path(args$log, paste(args$output, "log", sep = "//"))
write_hybrid_path(args$scale, paste(args$output, "scale", sep = "//"))
time_path <- paste(args$output, "time", sep = "//")
if (check_hybrid_path(time_path))
    elapsed_time["elapsed"] <-
        elapsed_time["elapsed"] + read_hybrid_path(time_path)
write_hybrid_path(elapsed_time["elapsed"], time_path)

cat("[Info] Done!\n")