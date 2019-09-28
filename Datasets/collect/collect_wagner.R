#! /usr/bin/env Rscript
# by wangshuai
# 13 May 2019
# 09:27:25 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#READ files
cat("Reading files...\n")
clusternames <- read.csv("../download/Wagner/GSE112294_ClusterNames.csv",header = T,stringsAsFactors = F)
clusternames <- clusternames[,2:3]

clusterid1 <- read.csv("../download/Wagner/GSM3067189_04hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid2 <- read.csv("../download/Wagner/GSM3067190_06hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid3 <- read.csv("../download/Wagner/GSM3067191_08hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid4 <- read.csv("../download/Wagner/GSM3067192_10hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid5 <- read.csv("../download/Wagner/GSM3067193_14hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid6 <- read.csv("../download/Wagner/GSM3067194_18hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid7 <- read.csv("../download/Wagner/GSM3067195_24hpf_clustID.txt.gz",stringsAsFactors = F,col.names = 'clusterid',header = F)
clusterid <- rbind(clusterid1,clusterid2,clusterid3,clusterid4,clusterid5,clusterid6,clusterid7)

exprs1 <- read.csv("../download/Wagner/GSM3067189_04hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs2 <- read.csv("../download/Wagner/GSM3067190_06hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs3 <- read.csv("../download/Wagner/GSM3067191_08hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs4 <- read.csv("../download/Wagner/GSM3067192_10hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs5 <- read.csv("../download/Wagner/GSM3067193_14hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs6 <- read.csv("../download/Wagner/GSM3067194_18hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)
exprs7 <- read.csv("../download/Wagner/GSM3067195_24hpf.csv.gz",header = T,row.names = 1,stringsAsFactors = F)

cellnames1 <- data.frame('cellnames' = colnames(exprs1))
cellnames2 <- data.frame('cellnames' = colnames(exprs2))
cellnames3 <- data.frame('cellnames' = colnames(exprs3))
cellnames4 <- data.frame('cellnames' = colnames(exprs4))
cellnames5 <- data.frame('cellnames' = colnames(exprs5))
cellnames6 <- data.frame('cellnames' = colnames(exprs6))
cellnames7 <- data.frame('cellnames' = colnames(exprs7))
cellnames <- rbind(cellnames1,cellnames2,cellnames3,cellnames4,cellnames5,cellnames6,cellnames7)

# create metadata
cat("Creating metadata...\n")
metadata <- cbind(cellnames,clusterid)
meta <- merge(metadata,clusternames,by.x='clusterid',by.y='ClusterID')
meta$lifestage = paste(substr(meta$ClusterName,1,3),'embryo',sep = ' ')
rownames(meta) <- meta$cellnames
meta$clusterid <- NULL; meta$cellnames <- NULL
colnames(meta) <- c('cell_type1','lifestage')

#READ DGE
cat("Reading DGE\n")
exprs <- cbind(exprs1,exprs2,exprs3,exprs4,exprs5,exprs6,exprs7)
inclued_cell <- intersect(row.names(meta),colnames(exprs))
exprs <- exprs[,inclued_cell]
meta <- meta[inclued_cell,]
exprs <- Matrix(as.matrix(exprs),sparse = T)

# Reading ontology
cell_ontology <- read.csv("../cell_ontology/zebrafish_embryo_cell_ontology.csv",
                          header = T,stringsAsFactors = F)
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

# Constructing dataset
cat("Constructing dataset\n")
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names=1)
construct_dataset("../data/Wagner", exprs, meta, datasets_meta, cell_ontology, y_low = 0.1)
message("Done!")