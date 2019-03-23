#! /usr/bin/env Rscript
# by caozj
# 24 Jan 2018
# 7:14:13 PM

# This script performs density clustering as described in
# Rodriguez, A. & Laio, A., 2014. Science

suppressPackageStartupMessages({
    library(argparse)
    library(rhdf5)
    library(densityClust)
})

parser <- ArgumentParser()
parser$add_argument("-x", dest = "x", type = "character", required = TRUE)
parser$add_argument("-c", dest = "cluster", type = "character",
                    required = TRUE)
parser$add_argument("-s", dest = "seed", type = "integer", default = NULL)
args <- parser$parse_args()

# args <- list(
#     "x" = "PCA/Biase/trial_0/result.h5//x",
#     "seed" = 123
# )

if (!is.null(args$seed)) {
    set.seed(args$seed)
}


#===============================================================================
#
#  Read data
#
#===============================================================================
cat("[Info] Reading data...\n")
x_split <- strsplit(args$x, "//")[[1]]
stopifnot(length(x_split) == 2)
x <- t(h5read(x_split[1], x_split[2]))
suppressWarnings(c <- as.integer(args$cluster))
if (is.na(c)) {
    c_split <- strsplit(args$cluster, "//")[[1]]
    c <- length(unique(h5read(c_split[1], c_split[2])))
}
cat(sprintf("[Info] Using cluster number = %d\n", c))


#===============================================================================
#
#  Compute rho and delta
#
#===============================================================================
cat("[Info] Performing density clustering...\n")
dc <- densityClust(dist(x), gaussian = TRUE)


#===============================================================================
#
#  Threshold based clustering
#
#===============================================================================

# graph_name <- file.path(dirname(x_split[1]), "densityClust_decision_graph.pdf")
# rho <- 0.5
# delta <- 0.5

# while (TRUE) {
#     pdf(graph_name)
#     dc <- findClusters(dc, rho = rho, delta = delta, plot = TRUE)
#     dev.off()
#     cat(sprintf("[Info] Please check for file %s\n", graph_name))

#     break_flag <- FALSE
#     while (TRUE) {
#         cat("Accept threshold? (y/n) ")
#         confirm <- readLines("stdin", n = 1)
#         if (confirm == "y") {
#             break_flag <- TRUE
#             break
#         } else if (confirm == "n") {
#             break_flag <- FALSE
#             break
#         }
#     }
#     if (break_flag) {
#         break
#     }

#     cat(sprintf("Enter new threshold for rho (currently %f): ", rho))
#     rho_str <- readLines("stdin", n = 1)
#     if (rho_str != "") {
#         rho <- as.double(rho_str)
#     }
#     cat(sprintf("Enter new threshold for delta (currently %f): ", delta))
#     delta_str <- readLines("stdin", n = 1)
#     if (delta_str != "") {
#         delta <- as.double(delta_str)
#     }
# }


#===============================================================================
#
#  Product ordering based clustering
#
#===============================================================================
peaks <- order(dc$rho * dc$delta, decreasing = TRUE)[1:c]
dc <- findClusters(dc, peaks = peaks)


#===============================================================================
#
#  Save results
#
#===============================================================================
file_content <- h5ls(x_split[1])
if (! "densityClust" %in% subset(
    file_content, (otype == "H5I_GROUP") & (group == "/"), "name", drop = TRUE
)) {
    h5createGroup(x_split[1], "densityClust")
}
existing_trials <- subset(file_content, group == "/densityClust",
                          "name", drop = TRUE)
existing_trials <- ifelse(
    length(existing_trials) == 0, -1,
    sapply(strsplit(existing_trials, "_"), FUN = function (x) {
        as.integer(x[2])
    })
)
new_trial <- max(existing_trials) + 1
h5write(dc$clusters, x_split[1], sprintf("densityClust/trial_%d", new_trial))

cat("[Info] Done!\n")
