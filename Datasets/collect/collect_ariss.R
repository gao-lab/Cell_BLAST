#! /usr/bin/env Rscript
# by wangshuai
# 11 Mar 2019
# 14:36:35 PM

suppressPackageStartupMessages({
  library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

#READ label file
cat("Reading label file...\n")
metadata1 <- read.table("../download/Ariss/wt_Rbf_and_populations.txt",header=T,stringsAsFactors = F)
row.names(metadata1) <- metadata1[,'CellName']
metadata1 <- metadata1[,c('Genotype','cell_type1')]

metadata2 <- read.table("../download/Ariss/Cells_and_population.txt",header=T,row.names=1,stringsAsFactors = F)
metadata2$Genotype <- 'wt'

includedcells<-union(row.names(metadata1),row.names(metadata2))
metadata <- rbind(metadata2,metadata1)
metadata <- metadata[which(row.names(metadata) %in% includedcells),]

celltypes <- read.csv('../download/celltypes',sep='\t')

metadata$lifestage <- 'third instar larva stage'
metadata$organ <- 'eye disc'
metadata$race <- 'Drosophila melanogaster'

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
included_cells <- intersect(rownames(metadata), colnames(expmerge))  
metadata <- metadata[included_cells, ] 
expmerge <- expmerge[, included_cells]
expmerge[is.na(expmerge)]<-0

expressed_genes <- rownames(expmerge)[rowSums(expmerge > 1) > 5]
expmerge <- Matrix(as.matrix(expmerge),sparse = T)

message("Constructing dataset...") 
dataset <- new("ExprDataSet",
               exprs = expmerge, obs = metadata,
               var = data.frame(row.names = rownames(expmerge)),
               uns = list(expressed_genes = expressed_genes) 
)

message("Saving data...") 
write_dataset(dataset, "../data/Ariss/data.h5")
cat("Done!\n")
