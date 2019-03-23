#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(ggsci)
})
source("../Utilities/utils.R")

df <- read.csv(
    "../Results/benchmark_time.csv",
    check.names = FALSE, stringsAsFactors = FALSE
)

if (Sys.getenv("methods") != "") {
    df$Method <- factor(df$Method, levels = strsplit(
        Sys.getenv("methods"), " "
    )[[1]])
    levels(df$Method) <- gsub("_", " ", levels(df$Method))
}  # Order in the environment variable determines the color order

color_mapping <- pal_d3("category10")(nlevels(df$Method))
names(color_mapping) <- levels(df$Method)

gp <- ggplot(data = df %>% group_by(Method, `Reference size`) %>% summarise(
    `Time per query (ms) SD` = sd(`Time per query (ms)`),
    `Time per query (ms)` = mean(`Time per query (ms)`)
), mapping = aes(
    x = `Reference size`, y = `Time per query (ms)`, col = `Method`,
    ymin = `Time per query (ms)` - `Time per query (ms) SD`,
    ymax = `Time per query (ms)` + `Time per query (ms) SD`
)) + geom_point() + geom_line(size = 1) + geom_errorbar(
    width = 0.07, size = 0.6
) + scale_x_log10() + scale_colour_manual(values = color_mapping)
ggsave("../Results/benchmark_time.pdf", mod_style(gp), width = 5.5, height = 4)
