#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(ggplot2)
    library(ggsci)
    library(dplyr)
})
source("../Utilities/utils.R")

# Read data
df <- read.csv(snakemake@input[["data"]], check.names = FALSE, stringsAsFactors = FALSE)
df$dataset <- factor(df$dataset, levels = df %>%
    select(dataset, n_cell) %>%
    arrange(n_cell) %>%
    distinct() %>%
    pull(dataset)
)  # This determines dataset order

# MAP
gp <- ggplot(data = df %>% group_by(dataset, genes, n_gene) %>% summarise(
    sd = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision)
), mapping = aes(
    x = n_gene, y = mean_average_precision,
    ymin = mean_average_precision - sd,
    ymax = mean_average_precision + sd,
    col = dataset
)) + geom_line() + geom_errorbar(width = 0.03) + scale_color_d3(
    name = "Dataset"
) + scale_x_log10(
    name = "Number of selected genes"
) + scale_y_continuous(
    name = "Mean average precision"
)
ggsave(snakemake@output[["map"]], mod_style(gp), width = 6, height = 3.5)
