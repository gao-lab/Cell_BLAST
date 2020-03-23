#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(jsonlite)
    library(ggplot2)
    library(reshape2)
    library(dplyr)
    library(extrafont)
})
source("../Utilities/utils.R")

snakemake@config[["method"]] <- c(
    snakemake@config[["gene_space_method"]],
    snakemake@config[["latent_space_method"]]
)

df <- read.csv(snakemake@input[["data"]], check.names = FALSE, stringsAsFactors = FALSE)
df$dataset <- factor(df$dataset, levels = df %>%
    select(dataset, n_cell) %>%
    arrange(n_cell) %>%
    distinct() %>%
    pull(dataset)
)  # This determines dataset order
levels(df$dataset) <- gsub("\\+", "\n", levels(df$dataset))
df$dimensionality <- factor(df$dimensionality)
df$rmbatch <- factor(df$rmbatch)
df$method <- factor(df$method, levels = snakemake@config[["method"]])
levels(df$method) <- gsub("[_.]", " ", levels(df$method))
df_summarize_seed <- df %>% group_by(method, dataset, dimensionality, rmbatch) %>% summarise(
    mean_average_precision = mean(mean_average_precision),
    seurat_alignment_score = mean(seurat_alignment_score)
)
color_mapping <- unlist(fromJSON(snakemake@input[["palette"]]))

gp <- ggplot(df %>% sample_frac(), aes(
    x = mean_average_precision, y = seurat_alignment_score,
    col = method, shape = dimensionality
)) + geom_point(alpha = 0.8, size = 2) + facet_wrap(~dataset, nrow = 2) + scale_shape_manual(
    values = c(0, 1, 2, 5, 6), na.value = 7,
    # values = c(15, 16, 17, 18, 3),
    name = "Dimensionality"
) + scale_color_manual(
    values = color_mapping, name = "Method"
) + scale_x_continuous(
    name = "Mean average precision"
) + scale_y_continuous(
    name = "Seurat alignment score"
) + theme(strip.text = element_text(size = 8))
ggsave(snakemake@output[["twoway"]], mod_style(gp), width = 6, height = 5.5)

gp <- ggplot(df %>% filter(method == "Cell BLAST") %>% sample_frac(), aes(
    x = mean_average_precision, y = seurat_alignment_score,
    col = rmbatch
)) + geom_point(alpha = 0.8, size = 2) + facet_wrap(~dataset, nrow = 2) + scale_color_discrete(
    name = expression(lambda[b])
) + scale_x_continuous(
    name = "Mean average precision"
) + scale_y_continuous(
    name = "Seurat alignment score"
) + theme(strip.text = element_text(size = 8))
ggsave(snakemake@output[["cb_elbow"]], mod_style(gp), width = 5, height = 5.5)

df_opt <- df %>% mutate(
    ms_sum = 0.9 * mean_average_precision + 0.1 * seurat_alignment_score
) %>% group_by(
    method, dataset, dimensionality, rmbatch
) %>% summarise(
    ms_sum = mean(ms_sum)
) %>% group_by(method) %>% summarize(
    dimensionality = dimensionality[which.max(ms_sum)],
    rmbatch = rmbatch[which.max(ms_sum)]
) %>% as.data.frame()

print(df_opt)

df <- merge(df, df_opt)
df_summarize_seed <- merge(df_summarize_seed, df_opt)

gp <- ggplot(df_summarize_seed, aes(
    x = mean_average_precision, y = seurat_alignment_score, col = method
)) + geom_point(size = 2.5, alpha = 0.8) + facet_wrap(~dataset, nrow = 2) + scale_color_manual(
    values = color_mapping, name = "Method"
) + scale_x_continuous(
    name = "Mean average precision"
) + scale_y_continuous(
    name = "Seurat alignment score"
) + theme(strip.text = element_text(size = 7))
ggsave(snakemake@output[["twowayopt"]], mod_style(gp), width = 6, height = 5.8)

# gp <- ggplot(df_summarize_seed, aes(
#     x = mean_average_precision, y = seurat_alignment_score,
#     col = method, shape = dataset
# )) + geom_point(
#     size = 2.5, alpha = 0.8
# ) + scale_color_manual(
#     name = "Method", values = color_mapping
# ) + scale_shape_manual(
#     name = "Dataset", values = c(15, 16, 17, 18)
# ) + scale_x_continuous(
#     name = "Mean average precision"
# ) + scale_y_continuous(
#     name = "Seurat alignment score"
# ) + guides(
#     shape = guide_legend(label.theme = element_text(size = 6), keyheight = 2)
# )
# ggsave(snakemake@output[["twowayopt"]], mod_style(gp), width = 6, height = 4.5)

for (this_dataset in unique(df$dataset)) {
    missing_methods <- setdiff(
        levels(df$method),
        unique(df %>% filter(dataset == this_dataset) %>% pull(method))
    )
    placeholder_df <- data.frame(
        method = missing_methods,
        dataset = rep(this_dataset, length(missing_methods)),
        dimensionality = rep(NA, length(missing_methods)),
        n_cell = rep(NA, length(missing_methods)),
        rmbatch = rep(NA, length(missing_methods)),
        seed = rep(NA, length(missing_methods)),
        time = rep(NA, length(missing_methods)),
        nearest_neighbor_accuracy = rep(-1, length(missing_methods)),
        mean_average_precision = rep(-1, length(missing_methods)),
        seurat_alignment_score = rep(-1, length(missing_methods)),
        batch_mixing_entropy = rep(-1, length(missing_methods)),
        check.names = FALSE
    )
    df <- rbind(df, placeholder_df[, colnames(df)])
}  # leaving out blank spaces for missing categories

df$missing_label <- ""
df[is.na(df$seed), "missing_label"] <- "N.A."

gp <- ggplot(df %>% group_by(method, dataset) %>% summarise(
    sd = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision),
    missing_label = unique(missing_label)
), aes(
    x = dataset, y = mean_average_precision, fill = method,
    ymin = mean_average_precision - sd,
    ymax = mean_average_precision + sd
)) + geom_bar(
    stat = "identity", position = position_dodge(0.9)
) + geom_errorbar(
    width = 0.3, position = position_dodge(0.9)
) + geom_text(
    aes(label = missing_label), fontface = "bold",
    y = 0.01, hjust = 0, angle = 90, size = 2.8, position = position_dodge(0.9)
) + scale_fill_manual(
    values = color_mapping, name = "Method"
) + scale_y_continuous(
    limits = c(0, 1), name = "Mean average precision"
) + scale_x_discrete(name = "Dataset")
ggsave(snakemake@output[["map"]], mod_style(gp), width = 6, height = 4.5)

gp <- ggplot(df %>% group_by(method, dataset) %>% summarise(
    sd = sd(seurat_alignment_score),
    seurat_alignment_score = mean(seurat_alignment_score),
    missing_label = unique(missing_label)
), aes(
    x = dataset, y = seurat_alignment_score, fill = method,
    ymin = seurat_alignment_score - sd,
    ymax = seurat_alignment_score + sd
)) + geom_bar(
    stat = "identity", position = position_dodge(0.9)
) + geom_errorbar(
    width = 0.3, position = position_dodge(0.9)
) + geom_text(
    aes(label = missing_label), fontface = "bold",
    y = 0.01, hjust = 0, angle = 90, size = 2.8, position = position_dodge(0.9)
) + scale_fill_manual(
    values = color_mapping, name = "Method"
) + scale_y_continuous(
    limits = c(0, 1), name = "Seurat alignment score"
) + scale_x_discrete(name= "Dataset")
ggsave(snakemake@output[["sas"]], mod_style(gp), width = 6, height = 4.5)
