#! usr/bin/env Rscript
# by weil
# this script collects all cellID and corresponded cell_type1 and CL in ACA_V1

source("../../Utilities/data.R", chdir = TRUE)
datasets <- dir("../ACA_datasets_V1/")
datasets_meta <- read.csv("../ACA_datasets.csv")

meta_all <- NULL
for (dataset in datasets) {
  cellID <- read_hybrid_path(paste0("../ACA_datasets_V1/", dataset, "/data.h5//obs_names"))
  meta <- read_hybrid_path(paste0("../ACA_datasets_V1/", dataset, "/data.h5//obs"))
  publication <- as.character(datasets_meta[datasets_meta$dataset_name == dataset, ]$publication)
  if(!is.null(meta$cell_type1)){
    meta_CL <- data.frame(cellID = cellID, 
                          cell_type1 = meta$cell_type1,
                          cell_ontology_class = meta$cell_ontology_class,
                          dataset = dataset,
                          publication = publication)
    meta_all <- rbind(meta_all, meta_CL)
  }
}
saveRDS(meta_all, "../ACA_cell_meta.rds")
