setwd("~/scDimRed-CaoZJ/Datasets/GEO_dataset/")
library(openxlsx)
gds_result <- read.table("gds_result_19-5-30.txt", sep = "\n", header = FALSE)

accession <- as.vector(gds_result[grep("Accession", gds_result[,1]), ])
accessions <- lapply(accession, function(x) strsplit(x, "\t")[[1]][3])
accessions <- lapply(accessions, function(x) strsplit(x, " ")[[1]][2])
accessions <- as.data.frame(accessions)
accessions <- t(accessions)

organism <- as.vector(gds_result[grep("Organism:", gds_result[,1]), ])
organisms <- lapply(organism, function(x) strsplit(x, "\t")[[1]][2])
organisms <- t(as.data.frame(organisms))


title <- gds_result[grep("Accession", gds_result[,1])+1, ]
titles <- data.frame()
titles[1,1] <- gds_result[1,1]
titles[2:length(title),1] <- title[1:length(title)-1]

gds_result_meta <- cbind(titles, accessions, organisms)
write.xlsx(gds_result_meta, "gds_result_meta_19-5-30.xlsx")
