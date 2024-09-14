# scPER2P

scPER2P: Parameter-Efficient Single-cell LLM for Translated Proteome Profiles

Protein expression levels are crucial for capturing the intricate dynamics of biological processes within cells. Recent advancements in protein sequencing technologies have enabled the simultaneous measurement of transcriptomics and proteomes. However, it is still associated with significant challenges high-throughput sequencing errors, precise quantifications, and high-quality antibodies.  To address those challenges, we introduce scPER2P, an end-to-end deep learning framework that translates single-cell RNA-seq data into proteome profiles. Our model includes single-cell language models and incorporates parameter-efficient fine-tuning techniques to facilitate proteome profile inference. Experimental results across multiple datasets reflect that scPER2P not only achieves high correlation coefficients and cosine similarities with true proteomic profiles but also maintains promising performance with significantly fewer parameters than the full fine-tuning methods. Additionally, cell type clustering results underscore the model's capability to significantly improve the accuracy of cell type annotation tasks. Our approach offers a promising solution to enhance and complement proteome profiling in single-cell studies.

Step 1: Installing environment packages

conda env create --file environment.yaml --name scPER2P

conda activate scPER2P

Step 2:Download datasets in data folder.

Go through and download datasets through datasetslink.

Step 3: Download scGPT model in model folder.

Go through and download the model as well as setting files through
model link.

Step
4: Modify the path configuration.

Step 5:  Run the demo
code.

python Demo.py
