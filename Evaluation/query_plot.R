#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(jsonlite)
    library(ggplot2)
    library(scales)
    library(reshape2)
    library(plyr)
    library(dplyr)
    library(MESS)
    library(extrafont)
    library(tools)
})
source("../Utilities/utils.R")


#===============================================================================
#
#  Preparation
#
#===============================================================================

# Read data
df <- read.csv(snakemake@input[["data"]], check.names = FALSE, stringsAsFactors = FALSE)
df$group <- factor(df$group, levels = df %>%
    distinct(group, ref_size) %>%
    arrange(ref_size) %>%
    pull(group)
)  # Determine order by reference size
levels(df$group) <- gsub("_", " ", levels(df$group))
for (method in snakemake@config[["method"]]) {
    snakemake@config[[gsub("_", " ", method)]] <- snakemake@config[[method]]
}  # Determine order by snakemake config order
df$method <- factor(df$method, levels = snakemake@config[["method"]])
levels(df$method) <- gsub("_", " ", levels(df$method))
snakemake@config[["method"]] <- gsub("_", " ", snakemake@config[["method"]])
color_mapping <- unlist(fromJSON(snakemake@input[["palette"]]))

# Preprocessing
df_summarize_seed <- df %>% group_by(method, group, threshold) %>% summarise(
    pos_mba = mean(pos_mba),
    neg_mba = mean(neg_mba),
    mba = mean(mba)
) %>% as.data.frame()

# Curve fitting
df_auc <- df %>% distinct(method, group)
df_fit <- list()
for (this_method in unique(df$method)) {
    for (this_group in unique(df$group)) {
        this_df <- df %>% filter(
            method == this_method, group == this_group
        ) %>% arrange(neg_mba)
        if (nrow(this_df) == 0) next

        df_auc[
            df_auc$method == this_method &
            df_auc$group == this_group,
            "auc"
        ] <- auc(
            x = c(0, this_df$neg_mba, 1),
            y = c(1, this_df$pos_mba, 0),
            type = "linear"
        )

        this_fit <- roc.smooth(
            this_df[, c("neg_mba", "pos_mba")],
            angle = - pi / 4, spar = 0.52, nknots = 9
        )
        this_fit$method <- this_method
        this_fit$group <- this_group
        df_fit[[length(df_fit) + 1]] <- this_fit
    }
}
df_fit <- Reduce(rbind, df_fit)
df_fit$method <- factor(df_fit$method, levels = levels(df$method))

# Select threshold
df_select <- df %>% group_by(method, threshold) %>% summarise(
    mba = median(mba)
) %>% group_by(method) %>% summarise(
    threshold = threshold[which.max(mba)]
) %>% as.data.frame()
df_select$type <- "optimal"
df_select <- rbind(data.frame(
    method = snakemake@config[["method"]],
    threshold = sapply(snakemake@config[["method"]], function(x) {
        ifelse(
            is.null(snakemake@config[[x]][["default"]]),
            NA, snakemake@config[[x]][["default"]]
        )
    }),
    type = rep("default", length(snakemake@config[["method"]]))
), df_select) %>% filter(
    !is.na(threshold)
) %>% distinct(
    method, .keep_all = TRUE
)
df_select$method <- factor(df_select$method, levels = snakemake@config[["method"]])
df_select$mt <- sprintf("%s (%s)", df_select$method, as.character(df_select$threshold))
df_select$mt <- factor(df_select$mt, levels = df_select %>% arrange(as.integer(method), threshold) %>% pull(mt))
write.csv(
    df_select %>% arrange(method),
    snakemake@output[["selected"]],
    row.names = FALSE
)


#===============================================================================
#
#  Plotting
#
#===============================================================================

# ROC
gp <- ggplot() + geom_path(
    data = df_fit,
    mapping = aes(
        x = neg_mba, y = pos_mba, col = method
    ), size = 1.5, alpha = 0.4
) + geom_point(
    data = df_summarize_seed,
    mapping = aes(
        x = neg_mba, y = pos_mba, col = method
    ), alpha = 0.7, stroke = 0, size = 2.0
) + scale_color_manual(
    name = "Method", values = color_mapping, limits = levels(df$method)
) + scale_x_continuous(
    name = "Negative type MBA", limits = c(-0.02, 1.05)
) + scale_y_continuous(
    name = "Positive type MBA", limits = c(-0.02, 1.05)
) + facet_wrap(~group, nrow = 2)
ggsave(snakemake@output[["roc"]], mod_style(gp), width = 6.0, height = 5.5)

# AUC
gp <- ggplot(data = df_auc, mapping = aes(
    x = method, y = auc, col = method, fill = method
)) + geom_boxplot(alpha = 0.3, width = 0.7) + scale_fill_manual(
    values = color_mapping, guide = FALSE
) + scale_color_manual(
    values = color_mapping, guide = FALSE
) + scale_x_discrete(
    name = "Method"
) + scale_y_continuous(
    name = "AUC"
)
ggsave(snakemake@output[["auc"]], mod_style(gp, rotate.x = TRUE), width = 4.5, height = 3.5)

# Average MBA at different cutoffs
gp <- ggplot(data = df %>% mutate(
    threshold = factor(threshold)
), mapping = aes(
    x = threshold, y = mba,
    fill = method, col = method
)) + geom_boxplot(alpha = 0.3) + facet_wrap(
    ~method, scales = "free_x"
) + scale_color_manual(
    name = "Method", values = color_mapping, guide = FALSE
) + scale_fill_manual(
    name = "Method", values = color_mapping, guide = FALSE
) + scale_x_discrete(
    name = "Threshold"
) + scale_y_continuous(
    name = "Average MBA"
) + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.6)
)
ggsave(snakemake@output[["pnsum"]], mod_style(gp), width = 5.5, height = 3.5)

# Preparing plots for default thresholds (for methods with no defaults, optimal is used)
df_select_default <- df_select %>% distinct(method, .keep_all = TRUE)
# mt_map_optimal <- setNames(df_select_default$mt, df_select_default$method)
# color_mapping_default <- color_mapping
# names(color_mapping_default) <- mt_map_optimal[names(color_mapping_default)]
# color_mapping_default <- color_mapping_default[!is.na(names(color_mapping_default))]

df_default <- merge(df, df_select_default)
df_summarize_seed_default <- merge(df_summarize_seed, df_select_default)


# 2-way plot
find_hull <- function(df) df[chull(df$neg_mba, df$pos_mba), ]
hull <- ddply(df_summarize_seed_default, "method", find_hull)
gp <- ggplot() + geom_polygon(
    data = hull,
    mapping = aes(
        x = neg_mba, y = pos_mba, col = method, fill = method
    ), alpha = 0.1, linetype = 2, show.legend = FALSE
) + geom_point(
    data = df_summarize_seed_default,
    mapping = aes(
        x = neg_mba, y = pos_mba, col = method, shape = group
    ), size = 3.0
) + scale_color_manual(
    name = "Method", values = color_mapping
) + scale_fill_manual(
    name = "Method", values = color_mapping
) + scale_shape_manual(
    name = "Dataset group", values = c(15, 16, 17, 18)
) + scale_x_continuous(
    name = "Negative type MBA"
) + scale_y_continuous(
    name = "Positive type MBA"
) + guides(
    color = guide_legend(order = 1),
    shape = guide_legend(order = 0)
)
ggsave(snakemake@output[["pnopt"]], mod_style(gp), width = 5.0, height = 3.5)

# gp <- ggplot(df_default, aes(
#     x = mt, y = mba, col = mt, fill = mt
# )) + geom_point(
#     aes(shape = group),
#     size = 1.5, position = position_jitter(width = 0.2)
# ) + geom_boxplot(
#     alpha = 0.2, width = 0.7, outlier.shape = NA
# ) + scale_x_discrete(
#     name = "Method"
# ) + scale_y_continuous(
#     name = "Mean balanced accuracy"
# ) + scale_color_manual(
#     name = "Method", values = color_mapping_default, guide = FALSE
# ) + scale_fill_manual(
#     name = "Method", values = color_mapping_default, guide = FALSE
# ) + scale_shape_manual(
#     name = "Organ", values = c(15, 16, 17, 18), guide = FALSE
# )
# ggsave(snakemake@output[["mba"]], mod_style(gp), width = 4.2, height = 2.8)

# Integrated MBA plot
df_default_melted <- melt(
    df_default, measure.vars = c("pos_mba", "neg_mba", "mba"),
    variable.name = "mba"
)
df_default_melted$mba <- factor(
    df_default_melted$mba,
    levels = c("pos_mba", "neg_mba", "mba"),
    labels = c(
        "Positive types",
        "Negative types",
        "Average"
    )
)

gp <- ggplot(df_default_melted, aes(
    x = mba, y = value, col = method, fill = method
)) + geom_point(
    aes(shape = group, group = method), size = 1.5,
    position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.85)
) + geom_boxplot(
    alpha = 0.2, position = position_dodge(0.85), width = 0.7,
    outlier.shape = NA
) + scale_x_discrete(
    name = "Group of query cell types"
) + scale_y_continuous(
    name = "Mean balanced accuracy"
) + scale_shape_manual(
    name = "Dataset group", values = c(15, 16, 17, 18)
) + scale_color_manual(
    name = "Method", values = color_mapping
) + scale_fill_manual(
    name = "Method", values = color_mapping
) + guides(
    color = guide_legend(order = 1),
    fill = guide_legend(order = 1),
    shape = guide_legend(order = 0)
)
ggsave(snakemake@output[["mba"]], mod_style(gp), width = 5.5, height = 3.0)
