#sc_desc.h5ad: 2_DESC result
#loacdesc.h5ad: train mnn result, include traindata/testdata/validation
#sc_test.h5ad: train mnn result, dims=[scdata.shape[1],celltype]

library(ggplot2)
library(Seurat)
library(SeuratDisk)
library(loomR)
library(SeuratData)
library(dplyr)
#用adata.obs构造，再将desc赋给obs
Convert("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno.h5ad", dest = "h5seurat", overwrite = TRUE)
desc_result_1 <- LoadH5Seurat("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno.h5seurat")
desc_result_1
head(desc_result_1)
head(colnames(desc_result_1))
head(rownames(desc_result_1))
head(desc_result_1$RNA@counts)
head(desc_result_1$RNA@data)
saveRDS(desc_result_1,file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno.rds")
desc_norm <- NormalizeData(desc_result_1, normalization.method = "LogNormalize")
saveRDS(desc_norm,file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm.rds")
head(desc_norm)
SaveH5Seurat(desc_norm, filename = "/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm.h5Seurat")
Convert("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm.h5Seurat", dest = "h5ad")

desc_norm=readRDS("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm.rds")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='desc_0.8',reduction="umap0.8",label=TRUE)
DimPlot(object = desc_norm, group.by ='orig.ident',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()
aa=table(desc_norm$desc_0.8)
write.csv(aa,"/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/desc_08.csv")
bb=table(desc_norm$desc_0.8,desc_norm$orig.ident)
write.csv(bb,"/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/orig.ident.csv")
max_1=apply(bb, 1, function(t) colnames(bb)[which.max(t)])  #输出最大概率对应的行名-celltype
write.csv(max_1,"/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/orig.ident_1.csv")


Convert("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_sex.h5ad", dest = "h5seurat", overwrite = TRUE)
desc <- LoadH5Seurat("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_sex.h5seurat")
desc
cc=table(desc_norm$desc_0.8,desc$SEX)
write.csv(cc,"/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sex.csv")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_sex.pdf",width=8,height=6)
DimPlot(object = desc, group.by ='SEX',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()



#markers
nephron=c("Cldn1","Spp2","Lrp2","Aqp1","Sptssb","Slc12a1","Slc12a3","Calb1")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/markers_nephron.pdf",height=20,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features =nephron ,reduction ="umap0.8",ncol=2 )#
while (!is.null(dev.list()))  dev.off()
ureteric=c("Hsd11b2","Aqp4","Aqp2","Atp6v1g3")  # ureteric epithelium输尿管
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/markers_ureteric.pdf",height=10,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features =ureteric ,reduction ="umap0.8",ncol=2 )#
while (!is.null(dev.list()))  dev.off()
vascular=c("Kdr","Cdh5")  #血管
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/markers_vascular.pdf",height=5,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features =vascular ,reduction ="umap0.8",ncol=2 )#
while (!is.null(dev.list()))  dev.off()
immune=c("C1qc","Thy1","Cd79a","Tyrobp")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/markers_immune.pdf",height=10,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features =immune ,reduction ="umap0.8",ncol=2 )#
while (!is.null(dev.list()))  dev.off()
interstitial=c("Cnn1","Dcn")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/markers_interstitial.pdf",height=5,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features =interstitial ,reduction ="umap0.8",ncol=2 )#
while (!is.null(dev.list()))  dev.off()

type=c("vascular","immune","vascular","nephron","nephron","ureteric","nephron","vascular","immune","ureteric","nephron","nephron","nephron","immune","ureteric","interstitial","nephron","nephron","immune","unknown","immune","ureteric","nephron","immune","vascular","nephron","immune","interstitial")
desc_norm@meta.data$celltype<-plyr::mapvalues(x = desc_norm@meta.data$desc_0.8, from = c(0:27), to = type)#
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_anno.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='celltype',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()

#注释
subtype=c("Vascular","Macrophage","Vascular","PT","DT","Principal","PT","Vascular","B_lymphocytes","Principal","DT","tl-LoH","tl-LoH","T_lymphocytes","Intercalated","VSMC","PT","PT","T_lymphocytes","unknown","NK_myeloid","Principal","PT","NK_myeloid","Vascular","RC","NK_myeloid","Fibroblast")
desc_norm@meta.data$subcelltype<-plyr::mapvalues(x = desc_norm@meta.data$desc_0.8, from = c(0:27), to = subtype)#
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_anno_1.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='subcelltype',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()

subcelltype1=c("Vascular_1","Macrophage","Vascular_2","PT_1","DT_1","Principal_1","PT_2","Vascular_3","B_lymphocytes","Principal_2","DT_2","tl-LoH_1","tl-LoH_2","T_lymphocytes","Intercalated","VSMC","PT_3","PT_4","T_lymph_2","unknown","NK_myeloid","Principal_3","PT_5","NK_myeloid_2","Vascular_4","RC","NK_myeloid_3","Fibroblast")
desc_norm@meta.data$subcelltype1<-plyr::mapvalues(x = desc_norm@meta.data$desc_0.8, from = c(0:27), to = subcelltype1)#
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_anno_2.pdf",width=9,height=6)
DimPlot(object = desc_norm, group.by ='subcelltype1',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()
saveRDS(desc_norm,"/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm_1.rds")
#DE
pbmc_small <- SetIdent(desc_norm,value='desc_0.8')
all.markers <- FindAllMarkers(object = pbmc_small, test.use = "wilcox")
head(x = all.markers)
write.table(all.markers, file = "/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/ctkidney_sc_desc.markers",sep='\t',quote=F)

## heatmap
all.markers=read.table("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/ctkidney_sc_desc.markers")
top <- all.markers  %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
imm_heat4=c(top$gene)
desc_norm$desc_0.8=droplevels(desc_norm$desc_0.8)
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/heatmap_desc08.pdf",height=30)
DoHeatmap(desc_norm, features = imm_heat4, group.by='desc_0.8',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
while (!is.null(dev.list()))  dev.off()
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/heatmap_desc08_1.pdf")
DoHeatmap(desc_norm, features = imm_heat4, group.by='desc_0.8',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
while (!is.null(dev.list()))  dev.off()
#celltype1
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/heatmap_celltype.pdf",height=30)
DoHeatmap(desc_norm, features = imm_heat4, group.by='celltype',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
DoHeatmap(desc_norm, features = imm_heat4, group.by='subcelltype',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
while (!is.null(dev.list()))  dev.off()


############################################################################
#wrong:Cannot find 'umap0.8' in this Seurat object
sc=readRDS("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/celltrek_kidney/Mouse_Adult_DGE.rds")
head(sc@assays$RNA@counts)
head(sc@assays$RNA@data)
counts=sc@assays$RNA@counts

head(colnames(counts))
head(rownames(counts))
meta_data = read.csv("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/2_desc/2_desc_222copy/dge/obs.csv") # load in meta_data (barcodes, clusters, and nUMI)
head(meta_data)
sc@meta.data$desc_0.8=meta_data$desc_0.8
head(sc@meta.data)
saveRDS(sc,file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno.rds")
desc_norm <- NormalizeData(sc, normalization.method = "LogNormalize")
saveRDS(desc_norm,file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/sc_desc_anno_norm.rds")
head(desc_norm)

pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='desc_0.8',reduction="umap0.8",label=TRUE)
DimPlot(object = desc_norm, group.by ='orig.ident',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()

#wrong
Convert("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/2_desc/2_desc_222copy/sc_desc.h5ad", dest = "h5seurat", overwrite = TRUE)

#sc_desc先经过复制rawdata
#wrong


aa=table(desc_norm$desc_0.8,desc_norm$CellType)
write.csv(aa,"desc_08_anno.csv")
max_1=apply(aa, 1, function(t) colnames(aa)[which.max(t)])  #输出最大概率对应的行名-celltype
write.csv(max_1,"desc_08_anno_1.csv")
'''
max_1
            0             1             2             3             4 
    "Oligo_2"       "Inh_4"   "Astro_AMY"     "Ext_L23"     "Ext_Pir" 
            5             6             7             8             9 
 "Ext_Thal_1"     "Ext_L56" "Ext_Hpc_DG2"   "Inh_Pvalb"       "OPC_1" 
           10            11            12            13            14 
     "Ext_L6"       "Micro"     "Inh_Vip"       "Unk_1" "Inh_Meis2_3" 
           15            16            17            18            19 
"Ext_Hpc_CA1" "Inh_Meis2_2"   "Ext_Unk_3"   "Ext_Amy_1" "Inh_Meis2_1" 
           20            21            22            23            24 
      "Inh_6" "Ext_Hpc_CA3" "Ext_ClauPyr"      "LowQ_2"   "Ext_Unk_2" 
           25            26            27            28            29 
"Inh_Meis2_4"        "Nb_2"      "LowQ_1"       "Unk_2"      "LowQ_1" '''

type=c("B_naive_1","B_naive_2","TfH_1","B_GC_1","B_mem_1","T_CD4+_1","T_CD4+_2","T_CD8+_1","TfH_1","B_Cycling","B_naive_3","T_CD4+_3","NK_1","B_plasma","T_CD8+_2","B_GC_2","Monocytes","TfH_2","Endo_1","TfH_3","B_mem_2","B_mem_3","T TIM3+ DN","pDC" )
desc_norm@meta.data$celltype1<-plyr::mapvalues(x = desc_norm@meta.data$desc_0.8, from = c(0:23), to = type)#
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_anno.pdf",width=12,height=6)
DimPlot(object = desc_norm, group.by ='CellType',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/dimplot_anno_1.pdf",width=9,height=6)
DimPlot(object = desc_norm, group.by ='celltype1',reduction="umap0.8",label=TRUE)
while (!is.null(dev.list()))  dev.off()

head(rownames(desc_norm))
head(colnames(desc_norm))

pbmc_small <- SetIdent(desc_norm,value='desc_0.8')
all.markers <- FindAllMarkers(object = pbmc_small, test.use = "wilcox")
head(x = all.markers)
write.table(all.markers, file = "/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/c2llymph_sc_desc.markers",sep='\t',quote=F)
#awk '$7==0&&$6<0.05&&$3>0 {print $8}' c2lbrain_sc_desc.markers > genes_sc_desc/a0

## heatmap
all.markers=read.table("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/6_DE/c2llymphsc_desc.markers")
top <- all.markers  %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
imm_heat4=c(top$gene)
desc_norm$desc_0.2=droplevels(desc_norm$desc_0.8)
pdf(file="heatmap_desc02.pdf",height=30)
DoHeatmap(desc_norm, features = imm_heat4, group.by='desc_0.8',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
while (!is.null(dev.list()))  dev.off()
#celltype1
pdf(file="heatmap_celltype1.pdf",height=30)
DoHeatmap(desc_norm, features = imm_heat4, group.by='celltype1',size=4.0,label=FALSE)+scale_fill_gradientn(colors = c("white", "blue", "red"))
while (!is.null(dev.list()))  dev.off()

#st plot celltype : 13,5,  0,  1,  14,  7,   15,  3,   25,  21,  20,  29,  27,   4,    16,   19,   26,   2,    6,    22,
cells6 <- WhichCells(desc_norm, idents = c(13,5,  0,  1,  14,  7,   15,  3,   25,  21,  20,  29,  27,   4,    16,   19,   26,   2,    6,    22))
subset <- subset(x = desc_norm, cells =cells6)
saveRDS(subset,file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/sc_test/subset.rds")
#markers
##1:no
im=c("Adgre1","Il10","Cd68","C1qb","Itgam","Cd3e","Cd79a","Flt3","Cd3d","Nkg7","Gzma","S100a8","Mzb1","Jchain","Col1a1","Epcam","Top2a", "Ccr2")
imm=c("Adgre1", "Cd68", "C1qb","Top2a", "Ccr2","Itgam","Cd79a","Jchain","Flt3","S100a8","Gzma","Cd4","Cd3d","Cd8a","Pdpn","Hopx","Sftpa1","Epcam","Scgb3a2","Foxj1","Col13a1","Col1a2","Col14a1","Acta2","Vwf","Pecam1","Nrp1","Car4","Ccl21a")
im1=c("Ccr2","Itgam","Adgre1","Il10","C1qb","Cd68","Flt3","Cd3d","Cd3e","Cd79a","Nkg7","Gzma","S100a8","Mzb1","Jchain","Col1a1","Epcam")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/markers1.pdf",height=25,width=10)#,height=6,width=48
FeaturePlot(desc_norm, features = c("C1qb","Flt3","Epcam","Top2a", "Ccr2","Col13a1","Col1a2","Col14a1","Nrp1","Car4"),reduction ="umap0.2",ncol=2 )#
while (!is.null(dev.list()))  dev.off()

DC=c("Flt3")
Macro=c("Il10","Cd68","C1qb","Itgam")
Neutrophil=c("S100a8")
NK=c("Cd3e","Nkg7","Gzma")
Fibroblast=c("Col1a1")
T=c("Cd3e")#,"Cd4","Cd8a"
B=c("Cd79a")
Epithelial=c("Epcam")#上皮
CD4=c("Cd4","Cd8a")
Mast=c ("Kit", "Cpa3")
Plasma =c("Cd79a","Mzb1","Ighg1","Sdc1","Jchain")#
Monocyte=c("S100a8")
Endothelial=c("Pecam1","Vwf")#内皮
FeaturePlot(desc_norm, features = T,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = B,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Plasma,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Fibroblast,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Epithelial,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = DC,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Macro,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = NK,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Neutrophil,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Endothelial,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Mast,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = CD4,reduction ="umap0.2" )#
FeaturePlot(desc_norm, features = Monocyte,reduction ="umap0.2" )#
VlnPlot(desc_norm, features = Endothelial,group.by = "desc_0.2", pt.size =0)

##2:Allen Cell Types Database
interneuron=c("Gad1")
excitatory_neuron=c("Slc17a7")
microglia=c("Tyrobp") CD45 AXL
astrocyte=c("Aqp4")
oligodendrocyte_precursor=c("Pdgfra")
oligodendrocyte=c("Opalin")
endothelial=c("Nostrin")

MICRO=c("Ptprc")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/markers.pdf")#,height=36,width=48,height=6,width=48
FeaturePlot(desc_norm, features = "Ptprc",reduction ="umap0.2" )
while (!is.null(dev.list()))  dev.off()


pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/markers2.pdf")#,height=36,width=48,height=6,width=48
FeaturePlot(desc_norm, features = interneuron,reduction ="umap0.2" )
FeaturePlot(desc_norm, features = excitatory_neuron,reduction ="umap0.2" )
#FeaturePlot(desc_norm, features = microglia,reduction ="umap0.2" )
FeaturePlot(desc_norm, features = astrocyte,reduction ="umap0.2" )
FeaturePlot(desc_norm, features = oligodendrocyte_precursor,reduction ="umap0.2" )
FeaturePlot(desc_norm, features = oligodendrocyte,reduction ="umap0.2" )
FeaturePlot(desc_norm, features = endothelial,reduction ="umap0.2" )
while (!is.null(dev.list()))  dev.off()

##3
Fibroblast=c("Matn1","Cnmd","Col9a3","Acan","Col2a1","Col11a2")



#######################################################################################
#基因名有错
Convert("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/sc_desc_new.h5ad", dest = "h5seurat", overwrite = TRUE)
desc_result <- LoadH5Seurat("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/sc_desc_new.h5seurat")
desc_result
saveRDS(desc_result,"/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/sc_desc_new.rds")
desc_norm <- NormalizeData(desc_result, normalization.method = "LogNormalize")
'''
genes=c("ITGAM","CD3E","CD4","CD8A","S100A8","NKG7")
genes2=c("C1QC","EPCAM","COL1A1","FCGR3","PECAM1","VWF")
genes3=c("PTPRC","CD14","CD33","ITGAX","FCGR3A","CD79A","PTPRC")
Plasma =c("IGHG1", "MZB1", "SDC1", "CD79A")
mast=c ("CST3", "KIT", "CPA3")
myeloid=c("CD163", "CD68","FCN1","APOBEC3A","THBS1","CD33","CD141")
#T
Th = c("CD3D", "CD3E", "CD40LG")
gdT	= c("TRDV2", "TRGV9","CD27","CD45RA")
Treg =c("CD4","CD25","FOXP3")#,"CD127","CD152","TGFB", "IL10", "IL12","FOXP3", "STAT5")
th1 = c( "IL2", "IL12", "IL18")
th2 = c("CCR4", "IL4", "IL5")
tfh=c("CXCR5")
Th = c("CD3D", "CD3E", "CD40LG")
Treg =c("CD4","CD25","FOXP3")
CytotoxicTcell=c("CD8A")
#Tfh = c("CXCR5","ICOS","PD1")
#Th1=c("CXCR3","TBX21")#,"T-bet","CD183",,"T-Bet"
#Th2=c("GATA3","PTGDR2") #"CD294","CRTH2",
#Th17=c("CCR6","CD196","RORC")
CD4=C("CD3E","CD4","CD8A")
Mast=c ("KIT", "CPA3")
Plasma =c("MZB1","CD79A","IGHG1","SDC1")
Monocyte=c("S100A8")
Epithelial=c("PECAM1","VWF")
Endothelial=c("EPCAM")
'''
#由于/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/sc_desc.h5ad转换失败
counts <-Matrix::readMM("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/sc_combine_counts_sparse.mtx")
dim(counts) #3193 40532
var = read.csv("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/dge/var.csv")
head(var)
rownames(counts)=make.unique(var$X, sep = "_")
meta_data = read.csv("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/dge/obs.csv") # load in meta_data (barcodes, clusters, and nUMI)
head(meta_data)
rownames(meta_data)=meta_data$X
colnames(counts)<- meta_data$X
head(colnames(counts))
head(rownames(counts))
sc=CreateSeuratObject(
  counts,
  project = "SeuratProject",
  assay = "RNA",
  names.field = 1,
  meta.data = meta_data)
saveRDS(sc,"/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/sc_combine_counts_sparse.rds")
head(sc$RNA@counts)
desc_norm=NormalizeData(sc)
pbmc_small <- SetIdent(desc_norm,value='desc_0.2')
all.markers <- FindAllMarkers(object = pbmc_small, test.use = "wilcox")
head(x = all.markers)
write.table(all.markers, file = "/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/c2lbrain_sc_combine.markers",sep='\t',quote=F)

#awk '$7==0&&$6<0.05&&$3>0 {print $8}' c2lbrain_sc_combine.markers > genes/a0
#https://metascape.org/gp/index.html#/main/step1 选小鼠，选express analysis
#newtype=c("Macro_1","Macro_2","TCell","BCell","DC","NKCell","Macro_3","Macro_4","Neutrophil","Plasma","Fibroblast","Macro_5","Epithelial","Macro_6")

#obsm
obsm = read.csv("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/dge/obsm.csv") # load in meta_data (barcodes, clusters, and nUMI)
head(obsm)
desc_norm@reductions=obsm
umap=read.csv("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/dge/uns/umap.csv")
head(umap)
#markers
Fibroblast=c("Matn1","Cnmd","Col9a3","Acan","Col2a1","Col11a2")
interneuron=c("GAD1")
excitatory_neuron=c("SLC17A7")
microglia=c("TYROBP")
astrocyte=c("AQP4")
oligodendrocyte_precursor=c("PDGFRA")
oligodendrocyte=c("OPALIN")
endothelial=c("NOSTRIN")

sc=readRDS("/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/sc_combine_counts_sparse.rds")
head(sc$RNA@counts)
desc_norm=NormalizeData(sc)
saveRDS(desc_norm,file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/sc_combine_counts_sparse_norm.rds")
head(desc_norm)

pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/dimplot.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='desc_0.2',reduction="umap0.2",label=TRUE)
while (!is.null(dev.list()))  dev.off()

imm=c("Cd68","Flt3","Il10")
pdf(file="/home/huggs/shiyi/SCST/RCTD/RCTD/c2l_brain/test/5_DE/featurePlot.pdf",height=10,width=10)
FeaturePlot(desc_norm, features = imm,reduction ="umap0.6",ncol = 2,max.cutoff='q90' )
while (!is.null(dev.list()))  dev.off()

#summary
#type=c("Macro_1","Macro_2","Macro_3","Bcell","Neutrophil","T_CD8+","DC","NKCell","T_CD4+","Macro_4","Fibroblast","Plasma","Macro_5","Macro_6","Macro_7","Epithelial_1","Epithelial_2","ILC2_1","Endothelial","ILC2_2","NA","Epithelial_3")
type=c("Macro_1","Macro_2","DC_1","Bcell","Neutrophil","T_CD8+","DC_2","NKcell","T_CD4+","Macro_3","Fibroblast","Plasma","DC_3","DC_4","Macro_4","Epithelial_1","Epithelial_2","ILC2_1","Endothelial","ILC2_2","NA","Epithelial_3")
desc_norm@meta.data$desc_0.6<-plyr::mapvalues(x = desc_norm@meta.data$desc_0.6, from = c(0:21), to = type)#

pdf(file="/s/f/shiyi/item1/task1/220712/3_DE/desc2566406_dim.pdf",width=8,height=6)
DimPlot(object = desc_norm, group.by ='desc_0.6',reduction="umap0.6",label=TRUE)
DimPlot(object = desc_norm, group.by ='batch',reduction="umap0.6",label=TRUE)
DimPlot(object = desc_norm, group.by ='oldlabel',reduction="umap0.6",label=TRUE)
while (!is.null(dev.list()))  dev.off()
saveRDS(desc_norm,file="/s/f/shiyi/item1/task1/220712/3_DE/desc_train25664_3.rds")

imm=c("Cd68","Flt3","Il10")
pdf(file="/s/f/shiyi/item1/task1/220712/3_DE/featurePlot.pdf",height=15,width=15)
FeaturePlot(desc_norm, features = imm,reduction ="umap0.6",ncol = 2,max.cutoff='q90' )
while (!is.null(dev.list()))  dev.off()

#mean of each cluster
library(Seurat)
library(dplyr)
library(scater)
library(SeuratDisk)
library(loomR)
library(SeuratData)
desc_norm=readRDS("/s/f/shiyi/item1/task1/220712/3_DE/desc_train25664_3.rds")
desc_norm
#nCount_RNA /nFeature_RNA desc_0.6平均??youwenti
aa=aggregate(x= desc_norm$nCount_RNA, by = list(desc_norm$desc_0.6), FUN = "mean")
write.csv(aa,"/s/f/shiyi/item1/task1/220712/3_DE/nCount_RNA_ave.csv")
bb=aggregate(x= desc_norm$nFeature_RNA, by = list(desc_norm$desc_0.6), FUN = "mean")
write.csv(bb,"/s/f/shiyi/item1/task1/220712/3_DE/nFeature_RNA_ave.csv")

pdf(file="nFeature.pdf",width = 10)
VlnPlot(desc_norm, features = c("nFeature_RNA"),group.by = "desc_0.6", pt.size =0)
while (!is.null(dev.list()))  dev.off()

pdf(file="/s/f/shiyi/item1/task1/220712/3_DE/featurePlot1.pdf",height=15,width=15)
FeaturePlot(desc_norm, features = 'Ccl6',reduction ="umap0.6",max.cutoff='q90' ,slot='counts')
while (!is.null(dev.list()))  dev.off()
