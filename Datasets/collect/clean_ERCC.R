#! /usr/bin/env Rscript
# by weil
# Jan 21, 2019
# 01:44 PM

source("../../Utilities/data.R", chdir = TRUE)
datasets <- dir("../data/")
#delete "Merged"
datasets <- datasets[-grep("Merged", datasets)]
ERCC_dataset <- list()
for (dataset in datasets){
  file_path <- file.path("../data", dataset)
  gene_name <- tryCatch(
    read_hybrid_path(paste0(file.path(file_path, "data.h5"), "//var_names")),
    error = function(e) {
      message("Error in reading dataset ", dataset)
      gene_name <- NULL
    }
  )
  ERCC <- grepl("ERCC[^0-9]", gene_name, perl = TRUE, ignore.case = TRUE)
  if (sum(ERCC) != 0){
    ERCC_dataset[[dataset]] <- gene_name[ERCC]
    print(dataset)
  }
}
