#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(argparse)
    library(covr)
})

parser <- ArgumentParser()
parser$add_argument("-s", "--source", dest = "source", type = "character", required = TRUE)
parser$add_argument("-t", "--test", dest = "test", type = "character", required = TRUE)
args <- parser$parse_args()

file_coverage(args$source, args$test)
