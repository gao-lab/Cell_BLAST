suppressPackageStartupMessages({
    library(scales)
    library(ggplot2)
})


# Shared style
mod_style <- function(
    gp, rotate.x = FALSE, log.x = FALSE, log.y = FALSE,
    axis.text.x.size = NULL
) {
    if (log.x)
        gp <- gp + scale_x_log10(
            breaks = trans_breaks("log10", function(x) 10 ^ x),
            labels = trans_format("log10", math_format(10 ^ .x))
        )
    if (log.y)
        gp <- gp + scale_y_log10(
            breaks = trans_breaks("log10", function(x) 10 ^ x),
            labels = trans_format("log10", math_format(10 ^ .x))
        )
    args <- list(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size=axis.text.x.size)
    )
    if (rotate.x)
        args$axis.text.x = element_text(angle = 20, vjust = 0.6)
    gp <- gp + do.call(theme, args)
    gp
}


rotate.df <- function(x, angle) {
    stopifnot(ncol(x) == 2)
    rot <- matrix(c(
        cos(angle), - sin(angle),
        sin(angle), cos(angle)
    ), byrow = TRUE, nrow = 2, ncol = 2)
    rot.x <- as.matrix(x) %*% rot
    colnames(rot.x) <- colnames(x)
    as.data.frame(rot.x)
}


rotated.fit <- function(
    formula, data, weights, method = lm,
    angle = 0, n = 1000, ...
) {
    # check args
    x.name <- all.vars(formula[[3]])
    y.name <- all.vars(formula[[2]])
    stopifnot(length(x.name) == 1, length(y.name) == 1)
    data <- data[, c(x.name, y.name)]

    # rotate
    data.rot <- rotate.df(data, angle)

    # fit
    if (missing(weights)) {
        fitted.curve <- method(formula, data.rot, ...)
    } else {
        print(weights)
        fitted.curve <- method(formula, data.rot, weights, ...)
    }

    # predict
    pred.rot <- data.frame(x = seq(
        from = min(data.rot[, x.name]),
        to = max(data.rot[, x.name]),
        length.out = n
    ))

    colnames(pred.rot) <- x.name
    pred.rot[, y.name] <- predict(fitted.curve, pred.rot)
    pred.rot <- pred.rot[order(pred.rot[, x.name]), ]

    # rotate back
    rotate.df(pred.rot, -angle)
}


rotated.spline <- function(
    data, angle, n = 1000, ...
) {  # Assuming first column as x and second column as y
    data.rot <- rotate.df(data, angle)
    spline <- smooth.spline(x = data.rot[, 1], y = data.rot[, 2],  ...)
    pred.rot.x <- seq(
        from = min(data.rot[, 1]),
        to = max(data.rot[, 1]),
        length.out = n
    )
    pred.rot.y <- predict(spline, pred.rot.x)$y
    pred.rot <- cbind(pred.rot.x, pred.rot.y)
    pred.rot <- pred.rot[order(pred.rot[, 1]), ]
    colnames(pred.rot) <- colnames(data)
    pred.rot <- as.data.frame(pred.rot)
    rotate.df(pred.rot, -angle)
}
