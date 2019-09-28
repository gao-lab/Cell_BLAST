#! /usr/bin/env Rscript
# by wangshuai
# 11 Mar 2019
# 14:36:35 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#Annotated code for both WT and rbf mutant cells

#READ metadata
cat("Reading metadata...\n")
#metadata1 <- read.table("../download/Ariss/wt_Rbf_and_populations.txt",header=T,stringsAsFactors = F)
#row.names(metadata1) <- metadata1[,'CellName']
#metadata1 <- metadata1[,c('Genotype','Population')]

metadata <- read.table("../download/Ariss/Cells_and_population.txt",header=T,row.names=1,stringsAsFactors = F)
#metadata$Genotype <- 'wt'

#includedcells<-union(row.names(metadata1),row.names(metadata2))
#metadata <- rbind(metadata2,metadata1)
#metadata <- metadata[which(row.names(metadata) %in% includedcells),]
colnames(metadata) <- 'cell_type1'

#metadata$lifestage <- 'third instar larva stage'


#READ DGE
cat("Reading DGE\n")
path <- "../download/Ariss/GSE115476"
fileNames <- dir(path) 
filePath <- sapply(fileNames, function(x){ 
  paste(path,x,sep='/')})   
data <- lapply(filePath, function(x){
  read.table(x, header=T,stringsAsFactors = F)})

i <- 1
for (name in names(data)){
  perfix<-substr(name,gregexpr(pattern = '_',text = name)[[1]]+1,gregexpr(pattern = "\\.",text = name)[[1]]-1)
  colnames(data[[i]]) <- lapply(colnames(data[[i]]),function(x){
    paste(perfix,x,sep='_')})
  genes <- data[[i]][,1]
  included_cells <- intersect(rownames(metadata), colnames(data[[i]]))
  data[[i]] <- data.frame(genes,data[[i]][, included_cells])
  i <- i+1
}

expmerge <- Reduce(function(x,y) merge(x,y,by=1,all=T),data)
row.names(expmerge)<-expmerge[,1]
expmerge<-expmerge[,-1]

#remove "other" cell type 
metadata <- metadata[metadata$cell_type1 != "other", ,drop=FALSE]
included_cells <- intersect(rownames(metadata), colnames(expmerge))  
metadata <- metadata[included_cells, ,drop=FALSE] 
expmerge <- expmerge[, included_cells]
expmerge[is.na(expmerge)]<-0

expmerge <- Matrix(as.matrix(expmerge),sparse = T)

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/drosophila_eye_disc_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names=1)
construct_dataset("../data/Ariss", expmerge, metadata, datasets_meta, cell_ontology, y_low = 0.1)
message("Done!")
