#!/usr/bin/env Rscript

source("../.Rprofile", chdir=TRUE)
suppressPackageStartupMessages({
    library(jsonlite)
    library(ggplot2)
    library(dplyr)
})
source("../Utilities/utils.R")

df <- read.csv(snakemake@input[["data"]], check.names = FALSE, stringsAsFactors = FALSE)
df$method <- factor(df$method, levels = snakemake@config[["method"]])
levels(df$method) <- gsub("_", " ", levels(df$method))
color_mapping <- unlist(fromJSON(snakemake@input[["palette"]]))

gp <- ggplot(data = df %>% group_by(method, size) %>% summarise(
    sd = sd(time), time = mean(time)
), mapping = aes(
    x = size, y = time, col = method,
    ymin = time - sd, ymax = time + sd
)) + geom_point() + geom_line(size = 1) + geom_errorbar(
    width = 0.07, size = 0.6
) + scale_x_log10(
    name = "Reference size"
) + scale_y_continuous(
    name = "Time per query cell (ms)"
) + scale_colour_manual(values = color_mapping, name = "Method")
ggsave(snakemake@output[[1]], mod_style(gp), width = 5.2, height = 3.0)
