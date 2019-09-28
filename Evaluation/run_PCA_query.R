#! /usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(RANN)
})
source("../Utilities/data.R")

N_EMPIRICAL <- 10000
MAJORITY_THRESHOLD <- 0.5

parser <- ArgumentParser()
parser$add_argument("-m", "--model", dest = "model", type = "character", required = TRUE)
parser$add_argument("-r", "--ref", dest = "ref", type = "character", required = TRUE)
parser$add_argument("-q", "--query", dest = "query", type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-a", "--annotation", dest = "annotation", type = "character", default = "cell_ontology_class")
parser$add_argument("--n-neighbors", dest = "n_neighbors", type = "integer", default = 10)
parser$add_argument("--min-hits", dest = "min_hits", type = "integer", default = 2)
parser$add_argument("-c", "--cutoff", dest = "cutoff", type = "double", nargs = "+", default = 0.1)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
parser$add_argument("--subsample-ref", dest = "subsample_ref", type = "integer", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()

cat("[Info] Loading model...\n")
genes <- read_hybrid_path(paste(args$model, "genes", sep = "//"))
rotation <- t(read_hybrid_path(paste(args$model, "rotation", sep = "//")))
do.log <- read_hybrid_path(paste(args$model, "log", sep = "//"))
do.scale <- read_hybrid_path(paste(args$model, "scale", sep = "//"))
if (do.scale)
    stop("Scaled PCA is not generalizable!")

cat("[Info] Processing reference data...\n")
ref <- read_dataset(args$ref)
ref <- normalize(ref)
if (!is.null(args$clean))
    ref <- clean_dataset(ref, args$clean)
ref <- ref[genes, ]
ref_exprs <- t(ref@exprs)
if (do.log)
    ref_exprs <- log1p(ref_exprs)
ref_latent <- ref_exprs %*% rotation
ref_label <- ref@obs[[args$annotation]]

cat("[Info] Building empirical distribution...\n")
set.seed(args$seed)
idx1 <- sample(nrow(ref_latent), N_EMPIRICAL, replace = TRUE)
idx2 <- sample(nrow(ref_latent), N_EMPIRICAL, replace = TRUE)
empirical <- sort(sqrt(rowSums(
    (ref_latent[idx1, ] - ref_latent[idx2, ]) ^ 2
)))

cat("[Info] Querying...\n")
query <- read_dataset(args$query)
query <- normalize(query)
if (!is.null(args$clean))
    query <- clean_dataset(query, args$clean)
query <- query[genes, ]
query_exprs <- t(query@exprs)
start_time <- proc.time()
if (do.log)
    query_exprs <- log1p(query_exprs)
query_latent <- query_exprs %*% rotation
nn.rs <- nn2(ref_latent, query_latent, k = args$n_neighbors)

time_per_cell <- NULL
pval <- findInterval(nn.rs$nn.dists, empirical) / length(empirical)
pval <- matrix(pval, nrow(nn.rs$nn.dists), ncol(nn.rs$nn.dists))
prediction_list <- list()
for (cutoff in args$cutoff) {
    prediction_list[[as.character(cutoff)]] <- rep("rejected", nrow(pval))
    for (i in 1:nrow(pval)) {
        count <- table(ref_label[nn.rs$nn.idx[i, pval[i, ] < cutoff]])
        total_count <- sum(count)
        if (total_count < args$min_hits)
              next
        argmax <- which.max(count)
        if (count[argmax] / total_count <= MAJORITY_THRESHOLD) {
            prediction_list[[as.character(cutoff)]][[i]] <- "ambiguous"
            next
        }
        prediction_list[[as.character(cutoff)]][[i]] <- names(count)[argmax]
    }
    if (is.null(time_per_cell))
        time_per_cell <- (proc.time() - start_time)["elapsed"] * 1000 / length(prediction_list[[as.character(cutoff)]])
}
cat(sprintf("[Info] Time per cell: %.3fms\n", time_per_cell))

cat("[Info] Saving results...\n")
if (file.exists(args$output))
    file.remove(args$output)
for (cutoff in names(prediction_list)) {
    write_hybrid_path(prediction_list[[cutoff]], sprintf("%s//prediction/%s", args$output, as.character(cutoff)))
}
write_hybrid_path(t(nn.rs$nn.idx), paste(args$output, "nni", sep = "//"))
write_hybrid_path(t(nn.rs$nn.dists), paste(args$output, "nnd", sep = "//"))
write_hybrid_path(t(pval), paste(args$output, "pval", sep = "//"))
write_hybrid_path(time_per_cell, paste(args$output, "time", sep = "//"))

message("[Info] Done!\n")
