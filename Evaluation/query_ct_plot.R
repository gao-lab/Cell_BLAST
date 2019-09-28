#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(readxl)
    library(dplyr)
    library(ggplot2)
    library(pheatmap)
    library(gridExtra)
    library(RColorBrewer)
    library(ggsci)
    library(extrafont)
})

breaks <- seq(0, 1, length.out = 50)
colors <- colorRampPalette(brewer.pal(n = 7, name = "YlOrRd"))(length(breaks))
plot_list <- list()
n_groups <- length(snakemake@config[["dataset_group"]])
for (idx in 1:n_groups) {
    group <- snakemake@config[["dataset_group"]][idx]
    df <- read_excel(
        snakemake@input[["data"]], sheet = group
    ) %>% as.data.frame() %>% arrange(desc(positive), desc(number))
    pos <- df %>% select(positive) %>% transmute(
        Type = factor(if_else(positive, "positive", "negative"), levels = c("positive", "negative"))
    )
    acc <- df %>% select(-cell_ontology_class, -positive, -number)
    clnum <- sprintf("%s (%d)", df$cell_ontology_class, df$number)
    rownames(pos) <- clnum
    rownames(acc) <- clnum
    colnames(acc) <- gsub("_", " ", colnames(acc))
    hm <- pheatmap(
        as.matrix(acc), color = colors, breaks = breaks,
        cluster_rows = FALSE, cluster_cols = FALSE,
        annotation_row = pos,  annotation_names_row = FALSE,
        annotation_colors = list(Type = c(
            positive = pal_d3("category10")(2)[1],
            negative = pal_d3("category10")(2)[2]
        )), gaps_row = sum(pos$Type == "positive"), border_color = NA,
        cellwidth = 10, main = gsub("_", " ", group),
        legend = idx == n_groups, annotation_legend = idx == n_groups,
        fontfamily = "Arial", silent = TRUE
    )
    plot_list[[group]] <- hm$gtable
}
pdf(NULL)
gp <- grid.arrange(arrangeGrob(grobs = plot_list, nrow = 1), newpage = FALSE)
ggsave(snakemake@output[[1]], gp, width = 22, height = 8)
