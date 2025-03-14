from functools import cache
# import transformers
# import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import math
from collections import Counter
from openai import OpenAI
import pdb
import os
import h5py
import json
import requests
import matplotlib.pyplot as plt
from scipy import sparse
import random
from memory import memory
import itertools
from scipy.interpolate import make_interp_spline

from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

import scCAD


class RareCellDetection:
    def __init__(self, model = "gpt", model_name="gpt-4o-mini"):
        self.model = model
        if model == "llama":
            model_id = "../meta-llama/Llama-3.1-70B-Instruct" # "meta-llama/Meta-Llama-3.1-70B-Instruct"
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        elif model == "gpt" or model == "prob":
            self.api_key = "sk-dgfU0q3uxvOoRDoR4c94Be540aFf4e178f468d88Ca1b3d5e"
            self.url = "https://api.gptplus5.com/v1"
            self.client = OpenAI(api_key=self.api_key, base_url=self.url)
        
        self.model_name = model_name
        
    def request_llm(self, prompt):
        self.url = "https://api.gptplus5.com/v1/chat/completions"
        payload = json.dumps({
            "messages": [
                {"role": "system",
                "content": "You are a professional medical scientist with abundant gene knowledge!"
                },
                {
                "role": "user",
                "content": prompt
                }
            ],
            "stream": False,
            "model": self.model_name,
            "temperature": 0.5,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_p": 1,
            "logprobs": True,
            "top_logprobs": 5
        })
        headers = {
            "Content-Type": "application/json",
            'Authorization': f'Bearer {self.api_key}',
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        data = json.loads(response.text)
        output = data['choices'][0]['message']['content']
        top_logprobs = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        return output, top_logprobs

    def llm_inference(self, prompt):
        if self.model == "llama":
            messages = [
                {"role": "system", "content": "You are a professional medical scientist with abundant gene knowledge!"},
                {"role": "user", "content": prompt},
            ]
        
            outputs = self.pipeline(
                messages,
                max_new_tokens=3000,
            )
            return outputs[0]["generated_text"][-1]['content']

        elif self.model == "gpt":
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                stream=False,
                messages=[
                    {"role": "system", "content": "You are a professional medical scientist with abundant gene knowledge!"},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content

        elif self.model == "prob":
            output, top_logprobs = self.request_llm(prompt)
            return str(top_logprobs)
        

    def TPM_modification(self,gene_matrix,species="human",id_type="gene_name",unknown_process="avg",cache_dir=""):
        # Empty strings for no cache
        if os.path.exists(cache_dir) and cache_dir!= "": # cache
            tpm_matrix=np.load(cache_dir)
            print("TPM matrix loaded from cache")
            return tpm_matrix
        
        if species=="human":
            gene_anno = pd.read_csv("data/ncbi_anno_human.tsv", sep='\t')
        elif species=="mouse":
            gene_anno = pd.read_csv("data/ncbi_anno_mouse.tsv", sep='\t')
        else:
            raise ValueError(f"Unsupported species: {species}")

        # 计算基因长度
        gene_anno["Gene Length"] = gene_anno.apply(
            lambda row: row["Protein length"] * 3 if not pd.isnull(row["Protein length"]) else row["End"] - row["Begin"],
            axis=1
        )

        default_gene_length = gene_anno['Gene Length'].mean()

        if id_type == "gene_name":
            gene_length_dict = gene_anno.set_index('Symbol')['Gene Length'].to_dict()
        elif id_type == "accession":
            gene_length_dict = gene_anno.set_index('Accession')['Gene Length'].to_dict()
        else:
            raise ValueError(f"Unsupported id_type: {id_type}")

        corrected_matrix = gene_matrix.copy()
        gene_lengths = corrected_matrix.columns.map(lambda gene: gene_length_dict.get(gene, default_gene_length))
        gene_lengths_df = pd.DataFrame(gene_lengths.values, index=corrected_matrix.columns, columns=['length'])
        # TPM = (gene_counts / gene_length * 1000) / sum(gene_counts / gene_length * 1000) * 1e6
        corrected_matrix = (corrected_matrix * 1000).div(gene_lengths_df['length'], axis=1)

        total_counts = corrected_matrix.sum(axis=0)
        tpm_matrix = corrected_matrix.div(total_counts, axis=1) * 1e6
        tpm_matrix = tpm_matrix.fillna(0)
        
        if (not os.path.exists(cache_dir)) and (not cache_dir=="") :
            np.save(cache_dir,tpm_matrix)
        print("TPM matrix caclulated done")
        return tpm_matrix

    def preprocess_dataset(self):
        if self.dataset == "Chung": # DO NOT NEED TPM IN THIS DATASET
            labels = pd.read_csv('./data/Chung_GSE75688_labels.txt', sep='\s+')
            matrix = pd.read_csv("./data/Chung_GSE75688_raw_TPM_matrix.txt", sep='\s+')
            
            selected_labels = labels[labels['type'] == 'SC']
            sc_samples = selected_labels['sample'].tolist()
            gene_titles = ["gene_id", "gene_name", "gene_type"]
            gene_info = matrix[[col for col in matrix.columns if col in gene_titles]]
            selected_matrix = matrix[[col for col in matrix.columns if col in sc_samples]]
            selected_matrix = selected_matrix.transpose()
            
            cell_union = [c for c in selected_labels['sample'] if c in selected_matrix.index]
            selected_matrix = selected_matrix.loc[cell_union]
            selected_labels.set_index('sample', inplace=True)
            selected_labels = selected_labels.loc[cell_union]
            
            data = np.array(selected_matrix) # Cells * Genes
            # data = np.vectorize(float)(data)
            labels = np.array(selected_labels['index2'])
            geneNames = np.array(gene_info['gene_name'])
            cellNames = np.array(selected_matrix.index)
            
            return data, labels, geneNames, cellNames

        elif self.dataset == "Darmanis":
            labels = pd.read_csv('./data/Darmanis_label.csv')
            matrix = pd.read_csv("./data/Darmanis_gene_expression_matrix.csv",index_col=0)
            data=self.TPM_modification(matrix,cache_dir="")
            data=np.array(data.T) 
            labels=np.array(labels['Sample_characteristics_ch1'])
            geneNames = np.array(np.array(matrix.index))
            cellNames = np.array(np.array(matrix.columns))

            return data,labels,geneNames,cellNames #  [466,22085]
            
        elif self.dataset == "Goolam":
            labels = pd.read_csv('./data/Goolam_label.csv')
            matrix = pd.read_csv("./data/Goolam_gene_expression_matrix.csv", index_col=0)
            data=self.TPM_modification(matrix,species="mouse",cache_dir="")
            data=np.array(data.T) # maybe float here?
            labels=np.array(labels.drop_duplicates()['tuple'])
            geneNames = np.array(np.array(matrix.index))
            cellNames = np.array(np.array(matrix.columns))

            new_labels = []
            for t in labels:
                if t == "('cleavage 2-cell', 'not applicable')":
                    tn = 'two'
                elif t == "('cleavage 8-cell', 'not applicable')":
                    tn = 'eight'
                elif t == "('cleavage 16-cell', 'not applicable')":
                    tn = 'sixteen'
                elif t == "('cleavage 32-cell', 'not applicable')":
                    tn = "thirtytwo"
                elif t == "('cleavage 4-cell', 'equatorial - equatorial')":
                    tn = 'fourEE'
                elif t == "('cleavage 4-cell', 'equatorial - meridional')":
                    tn = 'fourEM'
                elif t == "('cleavage 4-cell', 'meridional - equatorial')":
                    tn = 'fourME'
                elif t == "('cleavage 4-cell', 'meridional - meridional')":
                    tn = 'fourMM'
                new_labels.append(tn)
            new_labels = np.array(new_labels)
                        
            return data, new_labels, geneNames, cellNames

        elif self.dataset == "Jurkat":
            # Data matrix should only consist of values where rows represent cells and columns represent genes.
            data_mat = h5py.File('./data/1%Jurkat.h5')
            data = np.array(data_mat['X']).astype(int) # Cells * Genes
            labels = np.array(data_mat['Y'])
            geneNames = np.array(data_mat['gn'])
            cellNames = np.array(data_mat['cn'])
            data_mat.close()
            labels = np.array([str(i, 'UTF-8') for i in labels])
            geneNames = np.array([str(i, 'UTF-8') for i in geneNames])
            cellNames = np.array([str(i, 'UTF-8') for i in cellNames])
            df_data = pd.DataFrame(data, columns=geneNames, index=cellNames)
            df_data_tpm = self.TPM_modification(df_data,cache_dir="")
            new_labels = []
            for t in labels:
                if t == "jurkat":
                    tn = 'Jurkat'
                elif t == "293T":
                    tn = 'Tcell'
                new_labels.append(tn)
                
            return data, new_labels, geneNames, cellNames

        elif self.dataset == "Marsgt":
            gene_cell = sparse.load_npz('./data/Tutorial_example/RNA.npz')
            data = gene_cell.toarray().transpose()
            true_label = np.load('./data/Tutorial_example/label500.npy',allow_pickle=True)
            gene_names = pd.DataFrame(np.load('./data/Tutorial_example/gene_name.npy',allow_pickle=True)).transpose().values.tolist()
            cell_names = [i for i in range(len(true_label))]
            
            new_labels = []
            for t in true_label:
                if t == "CD4+ T naive":
                    tn = 'Tcell'
                elif t == "Plasma cell":
                    tn = 'Bcell'
                new_labels.append(tn)
            df_data = pd.DataFrame(data, columns=gene_names, index=cell_names)
            df_data_tpm=self.TPM_modification(df_data,cache_dir="")
            data=np.array(df_data_tpm)
            return data, new_labels, np.array(gene_names)[0], cell_names
        
        elif self.dataset == "Yang":
            labels = pd.read_csv('./data/Yang_label.txt', delimiter="\t")
            matrix = pd.read_csv("./data/Yang_data.csv", index_col=0)

            geneNames = np.array(np.array(matrix['Gene_id']))
            matrix.drop(columns=["Gene_id"], inplace=True)

            # 求交集
            common_samples = set(labels['sample_title']) & set(matrix.columns)
            # 过滤 labels 和 matrix
            matrix = matrix[list(common_samples)]
            labels = labels[labels['sample_title'].isin(common_samples)]
            labels = labels.set_index('sample_title').loc[matrix.columns].reset_index()
            labels = labels['cell_type']

            data=self.TPM_modification(matrix,species="mouse",cache_dir="")
            data=np.array(data.T)
            labels=np.array(labels)
            cellNames = np.array(np.array(matrix.columns))

            new_labels = []
            for t in labels:
                if t == "Ana6 matrix cells (TACs)" or t == "Ana6 matrix cells (TACs)":
                    tn = "Ana_matrix_cell"
                elif t == "Telogen hair germ cells":
                    tn = 'Telogen_hair_germ_cell'
                elif t == "Telogen bulge stem cells":
                    tn = 'Telogen_bulge_stem_cell'
                elif t == "Ana1 hair germ cells":
                    tn = "Ana_hair_germ_cell"
                try:
                    new_labels.append(tn)
                except:
                    print(t)
            new_labels = np.array(new_labels)

            return data, new_labels, geneNames, cellNames
    

        elif self.dataset == "MacParland":
            pdb.set_trace()
            labels = pd.read_csv('./data/MacParland_label.txt', delimiter="\t")
            matrix = pd.read_csv("./data/MacParland_exp.csv", index_col=0)
            
            geneNames = np.array(np.array(matrix.columns))

            # 求交集
            common_samples = set(labels['CellName']) & set(matrix.index)
            pdb.set_trace()
            # 过滤 labels 和 matrix
            matrix = matrix.loc[list(common_samples)]
            # labels = labels[labels['CellName'].isin(common_samples)]
            labels = labels.set_index('CellName').loc[matrix.columns].reset_index()
            labels = labels['CellType']

            data=self.TPM_modification(matrix,species="human",cache_dir="")
            labels=np.array(labels)
            cellNames = np.array(np.array(matrix.columns))
            return data, labels, geneNames, cellNames


        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

    def visualize(self, data, labels):
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

        # Reduce dimensionality with PCA before t-SNE for efficiency
        pca = PCA(n_components=50)
        data_pca = pca.fit_transform(data)
        # print(f"{self.dataset} 主成分方差占比：{[round(i, 4) for i in pca.explained_variance_ratio_]}")

        # Perform optimized t-SNE
        tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=500, random_state=42)
        tsne_results = tsne.fit_transform(data_pca)

        # Convert to DataFrame for visualization
        df_tsne = pd.DataFrame(tsne_results, columns=["t-SNE1", "t-SNE2"])
        df_tsne["Cell Type"] = labels

        # Generate colors for scatter plot
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("viridis", len(unique_labels))
        norm = mcolors.Normalize(vmin=0, vmax=len(unique_labels) - 1)

        # Plot t-SNE results
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(df_tsne["t-SNE1"], df_tsne["t-SNE2"], 
                            c=[np.where(unique_labels == label)[0][0] for label in df_tsne["Cell Type"]], 
                            cmap=cmap, norm=norm, alpha=0.7)

        # Highlight rare cell types
        for label in self.rare_types:
            rare_cells = df_tsne[df_tsne["Cell Type"] == label]
            plt.scatter(rare_cells["t-SNE1"], rare_cells["t-SNE2"], edgecolors='red', facecolors='none', s=100, label=label)

        # Create colorbar with specific labels
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label("Cell Types")
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)

        # Set title and labels
        plt.title(f"t-SNE Visualization of Gene Expression Dataset {self.dataset}")
        plt.xlabel("t-SNE1")
        plt.ylabel("t-SNE2")
        plt.legend(loc="upper right")  # Add legend for rare cell types
        plt.savefig(f"t-SNE-{self.dataset}.png")
    
    def rare_cluster_variance(self, data, labels):
        # 定义不同数据集的稀有细胞类型
        rare_types_dict = {
            "Darmanis": ["endothelial", "opc", "microglia"],
            "Chung": ["Stromal"],
            "Goolam": ["sixteen", "thirtytwo", "fourEE"],
            "Jurkat": ["Jurkat"],
            "Marsgt": ["Bcell"],
            "Yang": ['Telogen_bulge_stem_cell']
        }

        # 获取当前数据集的稀有类型
        rare_types = rare_types_dict.get(dataset, [])
        
        # 轮廓系数
        # 生成新的标签列表：将 rare_types 归为一类，其余不变
        rare_label = "rare"
        new_labels = np.array([rare_label if lbl in rare_types else lbl for lbl in labels])

        # 过滤仅包含 rare_types 的数据
        rare_mask = np.isin(labels, rare_types)
        rare_data = data[rare_mask]
        rare_labels = new_labels[rare_mask]  # 只保留 rare_types 的合并标签

        # 需要至少两个不同的类别，否则 silhouette_score 计算无效
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("细胞类型不足两个类别，无法计算轮廓系数")

        # 计算所有点的轮廓系数
        silhouette_vals = silhouette_samples(data, new_labels)

        # 选出 "rare" 细胞的轮廓系数
        rare_silhouette_vals = silhouette_vals[new_labels == rare_label]

        # 计算 rare 类的平均轮廓系数
        if len(rare_silhouette_vals) > 0:
            return np.mean(rare_silhouette_vals)
        else:
            return np.nan  # 如果 rare 细胞数目太少，返回 NaN

        # PCA variances
        '''
        # 筛选稀有细胞
        rare_indices = [i for i, label in enumerate(labels) if label in rare_types]
        rare_data = data[rare_indices]

        # 进行 PCA
        pca = PCA(n_components=5)
        data_pca = pca.fit_transform(rare_data)

        return pca.explained_variance_ratio_
        '''


    def generate_prompt(self, top_genes=[], relative_high_genes=None, relative_low_genes=None, tree=True, tree_high_genes=None, tree_low_genes=None):
        gene_list = ""
        for g in list(top_genes):
            gene_list += f" - {g} \n"

        if self.dataset == "Chung":
            sampling_info = "We performed single-cell RNA sequencing (RNA-seq) for 549 primary breast cancer cells and lymph node metastases from 11 patients with distinct molecular subtypes. We separated these single cells into epithelial tumor cells and tumor-infiltrating immune cells using inferred CNVs from RNA-seq. The refined single-cell profiles for tumor and immune cells provide key expression signatures of breast cancer and its surrounding microenvironment."
            cell_types = """ - Tumor: Primary breast cancer cells
            - Myeloid: Myeloid cells are a family of immune cells comprising monocytes, macrophages, myeloid dendritic cells (mDCs), granulocytes, and mast cells that originate from a common myeloid progenitor in the bone marrow.
            - Tcell: T cells are a type of white blood cell called lymphocytes. They help your immune system fight germs and protect you from disease.
            - Bcell: B cells are a type of white blood cell that makes infection-fighting proteins called antibodies.
            - Stromal: Stromal cells in the surrounding microenvironment
            """
            cell_types = """
            - Tumor: Primary breast cancer cells
            - Immune: A family of immune cells including Myeloid cells, T cells and B cells
            - Stromal: Stromal cells in the surrounding microenvironment
            """
            #cell_types = """
            #- Tumor: Primary breast cancer cells
            #- Immune: A family of immune cells including Myeloid cells, T cells and B cells
            #"""
            label_statement = "CellType should be one of 'Tumor', 'Myeloid', 'Bcell', 'Tcell', or 'Stromal'."
            label_statement = "CellType should be one of 'Tumor', 'Immune', or 'Stromal'."
            # label_statement = "CellType should be one of 'Tumor', 'Immune', or 'Unknown'."
            few_shots = memory['Chung']
        
        elif self.dataset == "Darmanis":
            sampling_info = "We used single cell RNA sequencing on 466 cells to capture the cellular complexity of the adult and fetal human brain at a whole transcriptome level. Healthy adult temporal lobe tissue was obtained from epileptic patients during temporal lobectomy for medically refractory seizures. We were able to classify individual cells into all of the major neuronal, glial, and vascular cell types in the brain."
            cell_types = """
            - neurons: Representing the major signaling cells of the brain.  
            - fetal_quiescent: Non-dividing fetal brain cells in a resting state.
            - hybrid: Cells with mixed characteristics of different types.  
            - astrocytes: Glial cells supporting neurons and maintaining the brain's environment.  
            - oligodendrocytes: Glial cells responsible for myelinating axons in the central nervous system.
            - fetal_replicating: Actively dividing cells in fetal brain development.
            - endothelial: Cells forming the lining of blood vessels in the brain.
            - opc: Oligodendrocyte Precursor Cells, Progenitor cells that can differentiate into oligodendrocytes.  
            - microglia: Immune cells of the brain involved in surveillance and response to injury or infection.
            """
            label_statement = "CellType should be one of 'neurons', 'fetal_quiescent', 'hybrid', 'astrocytes', 'oligodendrocytes', 'fetal_replicating', 'endothelial', 'opc', 'microglia'."
            few_shots = memory["Darmanis"]

        
        elif self.dataset == "Goolam":
            sampling_info = "Transcriptomes were determined for all blastomeres of 28 embryos at the 2- (n=8), 4- (n=16) and 8-cell (n=4) stages, and for individual cells taken from 16- (n=6) and 32- (n=6) cell stage embryos. We also carefully monitored the 2- to 4-cell divisions noting whether the cleavage plane was meridional (M, along the Animal-Vegetal (AV) axis marked by the second attached polar body) or equatorial (E, bisecting the AV axis) and the order in which such divisions occurred. This resulted in four groups of 4-cell stage embryos: ME, EM, MM and EE, which were all collected 10 hours after the first 2- to 4-cell division."
            cell_types = """
            - two: Developmental stage is cleavage 2-cell
            - fourEE: Developmental stage is cleavage 4-cell. Division pattern is equatorial - equatorial
            - fourEM: Developmental stage is cleavage 4-cell. Division pattern is equatorial - meridional
            - fourME: Developmental stage is cleavage 4-cell. Division pattern is meridional - equatorial
            - fourMM: Developmental stage is cleavage 4-cell. Division pattern is meridional - meridional
            - eight: Developmental stage is cleavage 8-cell.
            - sixteen: Developmental stage is cleavage 16-cell.
            - thirtytwo: Developmental stage is cleavage 32-cell.
            """
            label_statement = "CellType should be one of 'two', 'fourEE', 'fourEM', 'fourME', 'fourMM', 'eight', 'sixteen', 'thirtytwo'."
            few_shots = memory["Goolam"]
        
        elif self.dataset == "Jurkat":
            sampling_info = "The dataset consists of an equal-proportion in vitro mixture of 293T and Jurkat cells. Most of cells are T cells. Only very few cells are Tcell."
            cell_types = """
            - Jurkat: Human T-cell leukemia cells.
            - Tcell: Human Embryonic Kidney Cells.
            """
            label_statement = "CellType should be one of 'Jurkat' or 'Tcell'."
            few_shots = memory["Jurkat"]
        
        elif self.dataset == "Marsgt":
            sampling_info = "The dataset consists of two kinds of cells: Tcell and Bcell"
            cell_types = """
            - Tcell: CD4+ naive T cells
            - Bcell: Plasma cells
            """
            label_statement = "CellType should be one of 'Tcell' or 'Bcell'."
            few_shots = memory["Marsgt"] 
        
        elif self.dataset == "Yang":
            sampling_info = " "
            cell_types = """
            - Ana_matrix_cell: Ana matrix cells (TACs)
            - Telogen_hair_germ_cell: Telogen hair germ cells
            - Telogen_bulge_stem_cell: Telogen bulge stem cells
            - Ana_hair_germ_cell: Ana hair germ cells
            """
            label_statement = "CellType should be one of Ana_matrix_cell, Telogen_hair_germ_cell, Telogen_bulge_stem_cell, or Ana_hair_germ_cell"


        if relative_high_genes == None:
            prompt = f"""
            #### Task
            Analyze gene expression data to determine the most probable cell type based on the given descriptions.
            
            #### Sampling Information
            {sampling_info}
            
            #### Known Cell Types
            {cell_types}

            #### Genes Highly Expressed
            Identify genes that are highly expressed within the cell (with higher degrees of expression listed first).
            {top_genes}
            """

            chain_of_thoughts = """
            #### Steps for Analysis
            1. Recall the Impact of Each Gene: For each gene on the list of specifically highly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics.
            2. Determine the Cell Type: Use the impact of each gene and the cell descriptions provided to infer the most likely cell type. Cross-reference these features with the insights from step 1, focusing on genes that show distinct expression patterns. Draw connections between the expression of specific genes and the characteristics commonly observed in each cell type.
            3. Justify Your Conclusion: Present a well-supported rationale for your cell type classification. Clearly articulate how the combination of gene expression data and cell descriptions led to your conclusion. Highlight any key genes that were particularly informative in your classification and explain why they are significant for determining the cell type.
        
            #### Output Format
            After your reasoning, restate your cell type choice using the format:
            -<type: CellType>-
            """
        
        else:
            prompt = f"""
            #### Task
            Analyze gene expression data to determine the most probable cell type based on the given descriptions.
            
            #### Sampling Information
            {sampling_info}
            
            #### Known Cell Types
            {cell_types}

            #### Genes Highly Expressed
            Identify genes that are highly expressed within the cell (with higher degrees of expression listed first).
            {top_genes}

            #### Genes Especially Highly Expressed
            Identify genes that are especially expressed within the cell (higher than the average expression level).
            {relative_high_genes}  

            #### Genes Especially Lowly Expressed
            Identify genes that are Especially lowly expressed within the cell.
            {relative_low_genes}
            """

            chain_of_thoughts = """
            #### Steps for Analysis
            1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
            2. Determine the Cell Type: Use the impact of each gene and the cell descriptions provided to infer the most likely cell type. Cross-reference these features with the insights from step 1, focusing on genes that show distinct expression patterns. Draw connections between the expression of specific genes and the characteristics commonly observed in each cell type.
            3. Justify Your Conclusion: Present a well-supported rationale for your cell type classification. Clearly articulate how the combination of gene expression data and cell descriptions led to your conclusion. Highlight any key genes that were particularly informative in your classification and explain why they are significant for determining the cell type.
        
            #### Output Format
            After your reasoning, restate your cell type choice using the format:
            -<type: CellType>-
            """
            # one word
            chain_of_thoughts = """
            #### Steps for Analysis
            1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
            2. Determine the Cell Type: Use the impact of each gene and the cell descriptions provided to infer the most likely cell type. Cross-reference these features with the insights from step 1, focusing on genes that show distinct expression patterns. Draw connections between the expression of specific genes and the characteristics commonly observed in each cell type.
            3. Justify Your Conclusion: Present a well-supported rationale for your cell type classification. Clearly articulate how the combination of gene expression data and cell descriptions led to your conclusion. Highlight any key genes that were particularly informative in your classification and explain why they are significant for determining the cell type.
        
            #### Output Format
            Please just output one word to represent cell type. DO NOT include any more extra contents.
            """
        
        if tree:
            prompt = f"""
            #### Task
            Analyze gene expression data to determine the most probable cell type based on the given descriptions.
            
            #### Sampling Information
            {sampling_info}
            
            #### Known Cell Types
            {cell_types}

            #### Genes Especially Highly Expressed
            Identify genes that are especially expressed within the cell (higher than the average expression level).
            {tree_high_genes}  

            #### Genes Especially Lowly Expressed
            Identify genes that are Especially lowly expressed within the cell.
            {tree_low_genes}

            """
            chain_of_thoughts = """
            #### Steps for Analysis
            1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
            2. Determine the Cell Type: Use the impact of each gene and the cell descriptions provided to infer the most likely cell type. Cross-reference these features with the insights from step 1, focusing on genes that show distinct expression patterns. Draw connections between the expression of specific genes and the characteristics commonly observed in each cell type.
            3. Justify Your Conclusion: Present a well-supported rationale for your cell type classification. Clearly articulate how the combination of gene expression data and cell descriptions led to your conclusion. Highlight any key genes that were particularly informative in your classification and explain why they are significant for determining the cell type.
        
            #### Output Format
            Please just output one word to represent cell type. DO NOT include any more extra contents.
            """
            chain_of_thoughts = """
            #### Steps for Analysis
            1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
            2. Determine the Cell Type: Use the impact of each gene and the cell descriptions provided to infer the most likely cell type. Cross-reference these features with the insights from step 1, focusing on genes that show distinct expression patterns. Draw connections between the expression of specific genes and the characteristics commonly observed in each cell type.
            3. Justify Your Conclusion: Present a well-supported rationale for your cell type classification. Clearly articulate how the combination of gene expression data and cell descriptions led to your conclusion. Highlight any key genes that were particularly informative in your classification and explain why they are significant for determining the cell type.
        
            #### Output Format
            After your reasoning, restate your cell type choice using the format:
            -<type: CellType>-
            """

        prompt += chain_of_thoughts + label_statement
        return prompt


    def select_key_cells_by_rank(self, data):
        # data: gene expression level, Cells * Genes
        top_30_gene_indices_per_cell = np.argsort(data, axis=1)[:, -30:]
        top_30_gene_indices_per_cell = np.flip(top_30_gene_indices_per_cell, axis=1)
        column_means = np.mean(data, axis=0)
        relative_expression = data - column_means # 每行减去相应列的均值
        return top_30_gene_indices_per_cell, relative_expression
    
    def select_key_cells_by_prob(self, data, threshold=5):
        """
        Select the top 30 most important genes for each cell.

        Args:
            data (np.ndarray): Gene expression levels, shape (Cells, Genes).
            threshold (float): The threshold value for filtering gene expression levels.

        Returns:
            np.ndarray: Indices of the top 30 genes for each cell, shape (Cells, 30).
        """
        # Step 1: Threshold Filter
        filtered_data = np.where(data >= threshold, data, 0)

        # Step 2: Calculate Specificity Weight (Q) and Importance Probability (P)
        total_expression_per_gene = np.sum(filtered_data, axis=0)  # Sum of gene expression across all cells
        Q = np.divide(filtered_data, total_expression_per_gene, where=total_expression_per_gene != 0)  # Avoid division by zero

        total_expression_per_cell = np.sum(filtered_data, axis=1, keepdims=True)  # Sum of gene expression per cell
        P = np.divide(Q, total_expression_per_cell, where=total_expression_per_cell != 0)  # Avoid division by zero

        # Step 3: Select Top 30 Genes Based on Importance Probability
        top_30_gene_indices_per_cell = np.argsort(P, axis=1)[:, -30:]  # Indices of the top 30 genes
        top_30_gene_indices_per_cell = np.flip(top_30_gene_indices_per_cell, axis=1)  # Sort in descending order

        return top_30_gene_indices_per_cell, None
    
    def get_specific_genes(self, relative_cell_data, geneNames):
        """
        Args:
            relative_cell_data: 1D numpy array representing adjusted expression levels for each gene
            gene_info: pandas DataFrame with a column 'gene_name' containing gene information
        
        Returns:
            highly_gene_list: String of top 30 highly expressed genes
            lowly_gene_list: String of top 20 lowly expressed genes
        """
        # Get the indices that would sort the array in descending order
        ranked_indices = np.argsort(- relative_cell_data)  # Negative sign for descending order
        
        # Select top N highly expressed genes and top N lowly expressed genes
        highly_expressed_indices = ranked_indices[:30]  # Top 30
        lowly_expressed_indices = ranked_indices[-20:]  # Bottom 20
        
        # Map indices to gene names
        highly_genes = geneNames[highly_expressed_indices]
        lowly_genes = geneNames[lowly_expressed_indices]
        
        # Format the gene lists
        highly_gene_list = "".join(f" - {g} \n" for g in highly_genes)
        lowly_gene_list = "".join(f" - {g} \n" for g in lowly_genes)
        
        return highly_gene_list, lowly_gene_list


    def llm_reason(self, labels, geneNames, top_30_gene_indices_per_cell, relative_expression, exp_name):
        for i in tqdm(range(0, len(labels))):
            # Get the adjusted expression levels for the given cell
            top_genes = geneNames[top_30_gene_indices_per_cell[i]]

            relative_high_genes, relative_low_genes = None, None
            if type(relative_expression) != type(None):
                relative_cell_data = relative_expression[i]
                relative_high_genes, relative_low_genes = self.get_specific_genes(relative_cell_data, geneNames)

            os.makedirs(f"./prompt/{exp_name}/", exist_ok=True)
            os.makedirs(f"./exp/{exp_name}/", exist_ok=True)
            
            prompt = self.generate_prompt(top_genes, relative_high_genes, relative_low_genes)
            with open(f"./prompt/{exp_name}/{i}.txt", "w") as f:
                f.write(prompt)

            while True:
                try:
                    llmout = self.llm_inference(prompt)
                    break
                except Exception as e:
                    print("Inference Error:", e)
                    
            with open(f"./exp/{exp_name}/{i}.txt", "w") as f:
                f.write(llmout)
    

    def llm_reason_mr(self, labels, geneNames, top_30_gene_indices_per_cell, relative_expression, exp_name):
        def select_examples(labels, geneNames, top_30_gene_indices_per_cell):
            """随机选择每个细胞类型的一个 example."""
            example_dict = {}
            unique_labels = list(set(labels))
            
            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                example_idx = random.choice(indices)  # 随机选择一个细胞作为模板
                top_genes = geneNames[top_30_gene_indices_per_cell[example_idx]]

                relative_cell_data = relative_expression[example_idx]
                relative_high_genes, relative_low_genes = self.get_specific_genes(relative_cell_data, geneNames)
                example_dict[label] = [top_genes, relative_high_genes, relative_low_genes]
            
            return example_dict
        
        def generate_comparison_prompt(current_cell, example_cell, label):
            """构造 LLM 进行细胞与单个类型 example 对比和打分的 prompt."""
            if self.dataset == "Chung":
                sampling_info = "We performed single-cell RNA sequencing (RNA-seq) for 549 primary breast cancer cells and lymph node metastases from 11 patients with distinct molecular subtypes. We separated these single cells into epithelial tumor cells and tumor-infiltrating immune cells using inferred CNVs from RNA-seq. The refined single-cell profiles for tumor and immune cells provide key expression signatures of breast cancer and its surrounding microenvironment."
                cell_types = {
                "Tumor": "- Tumor: Primary breast cancer cells",
                "Immune": "- Immune: A family of immune cells including Myeloid cells, T cells and B cells",
                "Stromal": "- Stromal: Stromal cells in the surrounding microenvironment",
                }
            elif self.dataset == "Darmanis":
                sampling_info = "We used single cell RNA sequencing on 466 cells to capture the cellular complexity of the adult and fetal human brain at a whole transcriptome level. Healthy adult temporal lobe tissue was obtained from epileptic patients during temporal lobectomy for medically refractory seizures. We were able to classify individual cells into all of the major neuronal, glial, and vascular cell types in the brain."
                cell_types = {
                "neurons": "neurons: Representing the major signaling cells of the brain.",
                "fetal_quiescent": "fetal_quiescent: Non-dividing fetal brain cells in a resting state.",
                "hybrid": "hybrid: Cells with mixed characteristics of different types.",  
                "astrocytes": "astrocytes: Glial cells supporting neurons and maintaining the brain's environment.", 
                "oligodendrocytes": "oligodendrocytes: Glial cells responsible for myelinating axons in the central nervous system.",
                "fetal_replicating": "fetal_replicating: Actively dividing cells in fetal brain development.",
                "endothelial": "endothelial: Cells forming the lining of blood vessels in the brain.",
                "opc": "opc: Oligodendrocyte Precursor Cells, Progenitor cells that can differentiate into oligodendrocytes. ", 
                "microglia": "microglia: Immune cells of the brain involved in surveillance and response to injury or infection."
                }

            elif self.dataset == "Goolam":
                sampling_info = "Transcriptomes were determined for all blastomeres of 28 embryos at the 2- (n=8), 4- (n=16) and 8-cell (n=4) stages, and for individual cells taken from 16- (n=6) and 32- (n=6) cell stage embryos. We also carefully monitored the 2- to 4-cell divisions noting whether the cleavage plane was meridional (M, along the Animal-Vegetal (AV) axis marked by the second attached polar body) or equatorial (E, bisecting the AV axis) and the order in which such divisions occurred. This resulted in four groups of 4-cell stage embryos: ME, EM, MM and EE, which were all collected 10 hours after the first 2- to 4-cell division."
                cell_types = {
                "two": "two: Developmental stage is cleavage 2-cell",
                "fourEE": "fourEE: Developmental stage is cleavage 4-cell. Division pattern is equatorial - equatorial",
                "fourEM": "fourEM: Developmental stage is cleavage 4-cell. Division pattern is equatorial - meridional",
                "fourME": "fourME: Developmental stage is cleavage 4-cell. Division pattern is meridional - equatorial",
                "fourMM": "Developmental stage is cleavage 4-cell. Division pattern is meridional - meridional",
                "eight": "Developmental stage is cleavage 8-cell.",
                "sixteen": "Developmental stage is cleavage 16-cell.",
                "thirtytwo": "Developmental stage is cleavage 32-cell.",
                }
            
            elif self.dataset == "Marsgt":
                sampling_info = "The dataset consists of two kinds of cells: Tcell and Bcell"
                cell_types = {
                "Tcell": "Tcell: CD4+ naive T cells",
                "Bcell": "Bcell: Plasma cells",
                }

            elif self.dataset == "Yang":
                sampling_info = " "
                cell_types = {
                "Ana_matrix_cell": "Ana matrix cells (TACs)",
                "Telogen_hair_germ_cell": "Telogen hair germ cells",
                "Telogen_bulge_stem_cell": "Telogen bulge stem cells",
                "Ana_hair_germ_cell": "Ana hair germ cells",
                }
                label_statement = "CellType should be one of Ana_matrix_cell, Telogen_hair_germ_cell, Telogen_bulge_stem_cell, or Ana_hair_germ_cell"



            zero_shot = True
            if zero_shot:
                prompt = f"""
                #### Task
                Analyze gene expression data to determine whether the current cell is belonging to reference cell type based on the given descriptions.
                
                #### Sampling Information
                {sampling_info}
                
                #### Reference Cell Type
                {cell_types[label]}

                #### Genes Highly Expressed
                Identify genes that are highly expressed within the cell (with higher degrees of expression listed first).
                {current_cell[0]}

                #### Genes Especially Highly Expressed
                Identify genes that are especially expressed within the cell (higher than the average expression level).
                {current_cell[1]}  

                #### Genes Especially Lowly Expressed
                Identify genes that are Especially lowly expressed within the cell.
                {current_cell[2]}

                #### Steps for Analysis
                1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
                2. Evaluate the Cell Type: Use the impact of each gene and the cell descriptions provided to infer whether the current cell is belonging to reference cell type.
                3. Score: Mark a score 0 ~ 10, on how the current cell is belonging to reference type.

                #### Output Format
                Please just output one number between 0~10 to how the current cell is belonging to reference type. DO NOT include any more extra contents.
                """

            else:
                prompt = f"""
                #### Task
                Analyze gene expression data to determine whether the current cell is belonging to reference cell type based on the given descriptions.
                
                #### Sampling Information
                {sampling_info}
                
                #### Reference Cell Type
                {cell_types[label]}

                ################ An Example of Reference Cell Type
                #### Genes Highly Expressed
                Identify genes that are highly expressed within the cell (with higher degrees of expression listed first).
                {example_cell[0]}

                #### Genes Especially Highly Expressed
                Identify genes that are especially expressed within the cell (higher than the average expression level).
                {example_cell[1]}  

                #### Genes Especially Lowly Expressed
                Identify genes that are Especially lowly expressed within the cell.
                {example_cell[2]}
                ####################################

                ################ Current Cell
                #### Genes Highly Expressed
                Identify genes that are highly expressed within the cell (with higher degrees of expression listed first).
                {current_cell[0]}

                #### Genes Especially Highly Expressed
                Identify genes that are especially expressed within the cell (higher than the average expression level).
                {current_cell[1]}  

                #### Genes Especially Lowly Expressed
                Identify genes that are Especially lowly expressed within the cell.
                {current_cell[2]}
                ####################################

                #### Steps for Analysis
                1. Recall the Impact of Each Gene: For each gene on the list of highly expressed genes, specifically highly expressed genes, and specifically lowly expressed genes to provide a concise description of their biological functions, roles in cellular processes, and known associations with specific cell types or conditions. Highlight how the expression level of each gene may influence cellular behavior and characteristics, with a particular focus on genes with notably high or low expression.
                2. Evaluate the Cell Type: Use the impact of each gene and the cell descriptions provided to infer whether the current cell is belonging to reference cell type.
                3. Score: Mark a score 0 ~ 10, on how the current cell is belonging to reference type.

                #### Output Format
                Please just output one number between 0~10 to how the current cell is belonging to reference type. DO NOT include any more extra contents.
                """
            
            return prompt

        def parse_llm_output(llmout):
            """解析 LLM 生成的输出，提取得分."""
            try:
                score = int(''.join(filter(str.isdigit, llmout)))
            except:
                score = 0
            
            return score
        
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']
        
        # 选择每种细胞类型的 example
        example_dict = select_examples(labels, geneNames, top_30_gene_indices_per_cell)

        predictions = []
        for i in tqdm(range(len(labels))):
            top_genes = geneNames[top_30_gene_indices_per_cell[i]]
            
            relative_high_genes, relative_low_genes = None, None
            if relative_expression is not None:
                relative_cell_data = relative_expression[i]
                relative_high_genes, relative_low_genes = self.get_specific_genes(relative_cell_data, geneNames)
            current_cell = [top_genes, relative_high_genes, relative_low_genes]

            os.makedirs(f"./prompt/{exp_name}/", exist_ok=True)
            os.makedirs(f"./exp/{exp_name}/", exist_ok=True)
            
            llm_scores = {}
            for label, example_cell in example_dict.items():
                if self.OOD:
                    if label in self.rare_types:
                        continue

                # 为当前细胞与每个类型的 example 生成单独的对比 prompt
                prompt = generate_comparison_prompt(current_cell, example_cell, label)

                with open(f"./prompt/{exp_name}/{i}_{label}.txt", "w") as f:
                    f.write(prompt)

                while True:
                    try:
                        llmout = self.llm_inference(prompt)
                        break
                    except Exception as e:
                        print("Inference Error:", e)

                with open(f"./exp/{exp_name}/{i}_{label}.txt", "w") as f:
                    f.write(llmout)

                # 解析 LLM 评分结果
                score = parse_llm_output(llmout)
                llm_scores[label] = score

            # 根据评分结果确定当前细胞的分类
            predicted_label = max(llm_scores, key=llm_scores.get)
            predictions.append(predicted_label)

        return predictions

    
    def llm_reason_correlation(self, labels, geneNames, top_30_gene_indices_per_cell, relative_expression):        
        all_genes = []
        for i in tqdm(range(len(labels))):
            top_genes = geneNames[top_30_gene_indices_per_cell[i]]
            
            if relative_expression is not None:
                relative_cell_data = relative_expression[i]

                # Get the indices that would sort the array in descending order
                ranked_indices = np.argsort(- relative_cell_data)  # Negative sign for descending order
                
                # Select top N highly expressed genes and top N lowly expressed genes
                highly_expressed_indices = ranked_indices[:30]  # Top 30
                lowly_expressed_indices = ranked_indices[-20:]  # Bottom 20
                
                # Map indices to gene names
                highly_genes = geneNames[highly_expressed_indices]
                lowly_genes = geneNames[lowly_expressed_indices]

            current_cell = list(top_genes) + list(highly_genes) + list(lowly_genes)
            all_genes += current_cell

        def parse_llm_output(llmout):
            """解析 LLM 生成的输出，提取得分."""
            try:
                score = int(''.join(filter(str.isdigit, llmout)))
            except:
                score = 0
            
            return score

        def save_sorted_frequencies(all_genes, cell_type_labels, dataset):
            # 统计频率
            gene_counts = Counter(all_genes)
            cell_type_counts = Counter(cell_type_labels)
            
            # 按频率排序（降序）
            sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
            sorted_cell_types = sorted(cell_type_counts.items(), key=lambda x: x[1], reverse=True)

            high_freq_genes = [gene for gene, freq in sorted_genes if freq > 20]
            
            # 确保目录存在
            os.makedirs("./analysis", exist_ok=True)
            filepath = f"./analysis/{dataset}.txt"
            
            # 保存到文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("Genes and Frequencies:\n")
                for gene, freq in sorted_genes:
                    f.write(f"{gene}: {freq}\n")
                
                f.write("\nCell Types and Frequencies:\n")
                for cell_type, freq in sorted_cell_types:
                    f.write(f"{cell_type}: {freq}\n")

            print(f"Data saved to {filepath}")

            return high_freq_genes

        def generate_prompt(gene, cell_type):
            return f"""Please evaluate the correlation between gene {gene} and cell type {cell_type}. 
        Output only one numerical score (1, 2, or 3):
        1. Low correlation: The gene has no role or even a negative effect on the cell type's characteristics.
        2. Medium correlation: The gene positively influences the cell type's characteristics, but there is no direct link.
        3. High correlation: The gene's specific expression directly determines certain characteristics of the cell type."""

        def process_gene_cell_types(all_genes, cell_type_labels, dataset, llm_inference, parse_llm_output):
            # 去重
            unique_genes = list(set(all_genes))
            unique_cell_types = list(set(cell_type_labels))

            # 保存排序后的基因和细胞类型及其频率
            high_freq_genes = save_sorted_frequencies(all_genes, cell_type_labels, dataset)
            
            # 生成 gene-cell_type 组合
            gene_cell_type_pairs = list(itertools.product(high_freq_genes, unique_cell_types))

            # 结果存储文件
            llm_output_file = f"./analysis/{dataset}_llm_scores.txt"

            with open(llm_output_file, "w", encoding="utf-8") as f:
                for gene, cell_type in tqdm(gene_cell_type_pairs):
                    prompt = generate_prompt(gene, cell_type)
                    
                    # 调用 LLM 并处理异常
                    while True:
                        try:
                            llmout = llm_inference(prompt)
                            break
                        except Exception as e:
                            print("Inference Error:", e)
                    
                    # 解析 LLM 输出
                    score = parse_llm_output(llmout)
                    
                    # 写入文件
                    f.write(f"{gene}, {cell_type}, {score}\n")

            print(f"LLM outputs saved to {llm_output_file}")
        
        process_gene_cell_types(all_genes, labels, self.dataset, self.llm_inference, parse_llm_output)


    def llm_analyze_correlation(self, labels, geneNames, top_30_gene_indices_per_cell, relative_expression):

        def parse_analysis_file(filepath):
            """
            解析 analysis.txt，提取基因、细胞类型及其相关性分数。
            返回：cell_type_genes (按细胞类型分类的三类基因)
            """
            cell_type_genes = defaultdict(lambda: {1: set(), 2: set(), 3: set()})  # 每个细胞类型的基因分类

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(", ")
                    if len(parts) != 3:
                        continue  # 跳过无效行
                    
                    gene, cell_type, score = parts
                    score = int(score)
                    if score not in [1, 2, 3]:
                        score = 2
                    cell_type_genes[cell_type][score].add(gene)  # 存入对应分类

            return cell_type_genes

        # 解析 LLM 评分数据
        analysis_file = f"./analysis/{dataset}_llm_scores.txt"
        cell_type_genes = parse_analysis_file(analysis_file)
        
        cross_cell_gene_mapping = defaultdict(lambda: defaultdict(set))
        cross_cell_strong_correlation = defaultdict(int)
        
        for i in tqdm(range(len(labels))):
            top_genes = set(geneNames[top_30_gene_indices_per_cell[i]])
            
            if relative_expression is not None:
                relative_cell_data = relative_expression[i]
                ranked_indices = np.argsort(-relative_cell_data)  # 降序排序
                highly_expressed_indices = ranked_indices[:30]
                lowly_expressed_indices = ranked_indices[-20:]
                highly_genes = set(geneNames[highly_expressed_indices])
                lowly_genes = set(geneNames[lowly_expressed_indices])
                current_cell_genes = top_genes.union(highly_genes).union(lowly_genes)
            else:
                current_cell_genes = top_genes
            
            strong_related_genes = set()
            related_cells = set()
            
            for gene in current_cell_genes:
                for cell_type, genes_dict in cell_type_genes.items():
                    if gene in genes_dict[3]:  # 该基因在其他细胞类型中为强相关
                        strong_related_genes.add(gene)
                        related_cells.add(cell_type)
                        cross_cell_gene_mapping[labels[i]][cell_type].add(gene)
            
            cross_cell_strong_correlation[labels[i]] = len(strong_related_genes)
        
        # 计算所有 cell type 的强相关基因的平均数量
        avg_strong_related_genes = np.mean(list(cross_cell_strong_correlation.values()))
        
        # 存储交叉相关基因矩阵
        output_matrix_file = f"./analysis/{dataset}_gene_correlation_matrix.txt"
        with open(output_matrix_file, "w", encoding="utf-8") as f:
            f.write("Cell Type, Strongly Related Gene Count\n")
            for cell_type, count in cross_cell_strong_correlation.items():
                f.write(f"{cell_type}, {count}\n")
            f.write(f"\nAverage Strongly Related Genes: {avg_strong_related_genes:.2f}\n")
        print(f"Gene correlation matrix saved to {output_matrix_file}")
        
        # 存储详细的交叉细胞类型及其对应基因
        output_cross_cell_file = f"./analysis/{dataset}_cross_cell_genes.txt"
        with open(output_cross_cell_file, "w", encoding="utf-8") as f:
            for cell_type, related_dict in cross_cell_gene_mapping.items():
                f.write(f"Cell Type: {cell_type}\n")
                for related_cell, genes in related_dict.items():
                    f.write(f"  Related to {related_cell}: {len(genes)} genes\n")
                    f.write(f"  Genes: {', '.join(genes)}\n")
                f.write("\n")
        print(f"Cross-cell gene mapping saved to {output_cross_cell_file}")




    def llm_reason_tree(self, data, labels, geneNames, tree_nodes, exp_name):
        for i in tqdm(range(0, len(labels))):
            genes_expression = data[i, :]
            tree_high_genes = []  # 高于参考值的基因
            tree_low_genes = []  # 低于或等于参考值的基因
            
            for gene, threshold in tree_nodes:
                if gene in geneNames:
                    idx = list(geneNames).index(gene)
                    if genes_expression[idx] > threshold:
                        tree_high_genes.append(gene)
                    else:
                        tree_low_genes.append(gene)
            
            os.makedirs(f"./prompt/{exp_name}/", exist_ok=True)
            os.makedirs(f"./exp/{exp_name}/", exist_ok=True)
            
            prompt = self.generate_prompt(tree=True, tree_high_genes=tree_high_genes, tree_low_genes=tree_low_genes)
            with open(f"./prompt/{exp_name}/{i}.txt", "w") as f:
                f.write(prompt)

            while True:
                try:
                    llmout = self.llm_inference(prompt)
                    break
                except Exception as e:
                    print("Inference Error:", e)
                    
            with open(f"./exp/{exp_name}/{i}.txt", "w") as f:
                f.write(llmout)


    def compute_metrics(self, labels, exp_name):
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

        acc_tot = 0
        tp = 0
        pred_pos = 0
        gt_pos = 0
        for i in tqdm(range(len(labels))):
            with open(f"./exp/{exp_name}/{i}.txt") as f:
                llmout = f.read()
                
            # read output
            pattern = r"type: \w+"
            label = labels[i]
            try:
                prediction = re.search(pattern=pattern, string=llmout).group(0)[6:]
                #prediction = llmout
                #print(i, label, prediction)
            except:
                print("Wrong Format: ", i)
            
        
            if prediction == label:
                acc_tot += 1
                if prediction in self.rare_types:
                    tp += 1
            if prediction in self.rare_types:
                pred_pos += 1
            if label in self.rare_types:
                gt_pos += 1
            
            try:
                tot = i + 1
                acc = acc_tot / tot
                print(acc)
                
                precision = tp / pred_pos
                recall = tp / gt_pos
                f1 = 2 * precision * recall / (precision + recall)
                spec = (acc_tot - tp) / (tot - gt_pos)
                Gmean = math.sqrt(spec * recall)
                print(f"acc:{acc}, precision:{precision}, recall:{recall}, F1:{f1}, Gmean:{Gmean}, gt:{gt_pos}")
            except Exception as e:
                print(e)
                pass
    
    def compute_metrics_cls(self, labels, exp_name):
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

        predictions = []
        correct_labels = []
        
        for i in tqdm(range(len(labels))):
            with open(f"./exp/{exp_name}/{i}.txt") as f:
                llmout = f.read()
            
            # 提取 LLM 预测的类型
            pattern = r"type: (\w+)"
            label = labels[i]
            
            try:
                prediction = re.search(pattern, llmout).group(1)  # 提取类别名称
            except AttributeError:
                print("Wrong Format: ", i)
                prediction = "Unknown"  # 如果提取失败，给一个默认类别
            
            predictions.append(prediction)
            correct_labels.append(label)

        # 计算 Precision, Recall, Weighted F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, predictions, average='weighted')

        res = {
            "precision": precision,
            "recall": recall,
            "weighted_f1": f1
        }
        print(res)

    def compute_prob(self, labels, exp_name):
        prob_matrix = []
        for i in tqdm(range(len(labels))):
            with open(f"./exp/{exp_name}/{i}.txt") as f:
                llmout = f.read()

                prob_list = [item['logprob'] for item in eval(llmout)][:2]
                prob_matrix.append(prob_list)
        prob_matrix = np.array(prob_matrix).transpose()

        # 提取 X 和 Y 坐标
        x = prob_matrix[0]
        y = prob_matrix[1]

        # 高对比度颜色列表
        custom_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]  # 红、绿、蓝、黄、紫

        # 根据 labels 分配自定义颜色
        # 为字符串 labels 分配颜色索引
        unique_labels = list(set(labels))  # 获取唯一类别
        label_to_color = {label: idx for idx, label in enumerate(unique_labels)}  # 映射到颜色索引
        colors = [custom_colors[label_to_color[label]] for label in labels]

        # 绘制散点图，使用自定义颜色
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=colors, s=10, alpha=0.9)

        # 添加图例
        for label, color in zip(unique_labels, custom_colors):
            plt.scatter([], [], c=color, label=label, s=120)
        plt.legend(title="Labels", fontsize=12)

        # 保存图像
        output_filename = "scatter_plot_custom_high_contrast_colors.png"
        plt.savefig(output_filename, dpi=300)
        plt.close()

        print(f"散点图已保存为 {output_filename}")


        pth = [-0.1, -0.01, -0.001, -0.0001, -0.00001, -1e-6]
        for p in pth:
            predicted_ood = prob_matrix[0] < p  # 预测是否为OOD
            ground_truth_ood = labels == "Stromal"  # 实际是否为OOD

            # 计算评估指标
            acc = accuracy_score(ground_truth_ood, predicted_ood)
            precision = precision_score(ground_truth_ood, predicted_ood)
            recall = recall_score(ground_truth_ood, predicted_ood)
            f1 = f1_score(ground_truth_ood, predicted_ood)
            print("Metrics: ", p, acc, precision, recall, f1)

    def compute_metrics_mr(self, true_labels, exp_name):
        """计算分类评估指标，并从文件中读取结果进行评估."""
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']
            
        def parse_llm_output(llmout):
            """解析 LLM 生成的输出，提取得分."""
            try:
                score = int(''.join(filter(str.isdigit, llmout)))
            except:
                score = 0
            
            return score
        
        predicted_labels = []
        binary_true_labels = []
        binary_predicted_labels = []
        
        # 读取预测结果
        for i in range(len(true_labels)):
            llm_scores = {}
            for label in set(true_labels):
                exp_file = f"./exp/{exp_name}/{i}_{label}.txt"
                if os.path.exists(exp_file):
                    with open(exp_file, "r") as f:
                        llmout = f.read().strip()
                    llm_scores[label] = parse_llm_output(llmout)
            
            if llm_scores:
                if self.OOD:
                    predicted_label = max(llm_scores, key=llm_scores.get)
                    predicted_labels.append(predicted_label)
                    
                    binary_true_labels.append(1 if true_labels[i] in self.rare_types else 0)

                    largest_two_values = sorted(llm_scores.values(), reverse=True)[:2]
                    difference = largest_two_values[0] - largest_two_values[1]
                    threshold = 1
                    binary_predicted_labels.append(1 if difference < threshold else 0)
                else:
                    predicted_label = max(llm_scores, key=llm_scores.get)
                    predicted_labels.append(predicted_label)
                    binary_true_labels.append(1 if true_labels[i] in self.rare_types else 0)
                    binary_predicted_labels.append(1 if predicted_label in self.rare_types else 0)
                # print(binary_true_labels[-1], binary_predicted_labels[-1])
        
        # accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(binary_true_labels, binary_predicted_labels)
        recall = recall_score(binary_true_labels, binary_predicted_labels)
        f1 = f1_score(binary_true_labels, binary_predicted_labels)

        res = {
            # "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        print(res)


    def analysis_mr_confidence(self, true_labels, exp_name):
        """计算分类评估指标，并计算困惑度统计，包括稀有类型的平均困惑度."""

        # Define rare types for each dataset
        dataset_rare_types = {
            "Darmanis": ["endothelial", "opc", "microglia"],
            "Chung": ["Stromal"],
            "Goolam": ["sixteen", "thirtytwo", "fourEE"],
            "Jurkat": ["Jurkat"],
            "Marsgt": ["Bcell"],
            "Yang": ['Telogen_bulge_stem_cell']
        }
        self.rare_types = dataset_rare_types.get(self.dataset, [])

        def parse_llm_output(llmout):
            """解析 LLM 生成的输出，提取得分."""
            try:
                score = int(''.join(filter(str.isdigit, llmout)))
            except:
                score = 0
            return score

        predicted_labels = []
        binary_true_labels = []
        binary_predicted_labels = []

        # 存储每个样本的困惑度
        sample_perplexity = {}

        # 读取预测结果
        for i in range(len(true_labels)):
            llm_scores = {}
            for label in set(true_labels):
                exp_file = f"./exp/{exp_name}/{i}_{label}.txt"
                if os.path.exists(exp_file):
                    with open(exp_file, "r") as f:
                        llmout = f.read().strip()
                    llm_scores[label] = parse_llm_output(llmout)

            if llm_scores:
                # 计算每个样本的困惑度 (标准差)
                k = 2
                top_k_llm_scores = sorted(llm_scores.values(), reverse=True)[:k]
                perplexity = np.std(list(top_k_llm_scores))
                sample_perplexity[i] = perplexity

                if self.OOD:
                    predicted_label = max(llm_scores, key=llm_scores.get)
                    predicted_labels.append(predicted_label)
                    
                    binary_true_labels.append(1 if true_labels[i] in self.rare_types else 0)

                    largest_two_values = sorted(llm_scores.values(), reverse=True)[:2]
                    difference = largest_two_values[0] - largest_two_values[1]
                    threshold = 1
                    binary_predicted_labels.append(1 if difference < threshold else 0)
                else:
                    predicted_label = max(llm_scores, key=llm_scores.get)
                    predicted_labels.append(predicted_label)
                    binary_true_labels.append(1 if true_labels[i] in self.rare_types else 0)
                    binary_predicted_labels.append(1 if predicted_label in self.rare_types else 0)
                # print(binary_true_labels[-1], binary_predicted_labels[-1])

        # 计算每类样本的平均困惑度
        class_perplexity = defaultdict(list)
        perplexity_values = []

        for i, label in enumerate(true_labels):
            if i in sample_perplexity:
                class_perplexity[label].append(sample_perplexity[i])
                perplexity_values.append(sample_perplexity[i])

        class_avg_perplexity = {cls: np.mean(perplexities) for cls, perplexities in class_perplexity.items()}
        
        # 计算稀有类型的平均困惑度
        avg_perplexity = np.mean(perplexity_values) if perplexity_values else 0

        precision = precision_score(binary_true_labels, binary_predicted_labels)
        recall = recall_score(binary_true_labels, binary_predicted_labels)
        f1 = f1_score(binary_true_labels, binary_predicted_labels)
        # precision = precision_score(true_labels, predicted_labels, average="weighted")
        # recall = recall_score(true_labels, predicted_labels, average="weighted")
        # f1 = f1_score(true_labels, predicted_labels, average="weighted")

        res = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "rare_avg_perplexity": avg_perplexity  # 稀有类型的平均困惑度
        }

        print(f"{self.dataset} avg_confidence: {avg_perplexity}")
        print(class_avg_perplexity)

        # 画柱状统计图展示每类样本困惑度的平均值
        plt.figure(figsize=(10, 6))
        plt.bar(class_avg_perplexity.keys(), class_avg_perplexity.values(), color='skyblue', label="All Classes")
        
        # 标注稀有类型的平均困惑度
        if avg_perplexity > 0:
            plt.axhline(y=avg_perplexity, color='r', linestyle='--', label="Average Confidence")

        plt.xlabel("Sample Class")
        plt.ylabel("Average Perplexity (std of LLM scores)")
        plt.title("Average Perplexity per Sample Class")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.savefig(f"Perplexity {self.dataset}")


    def tree100(self, data, labels, geneNames, cellNames):
        # 根据数据集定义稀有类别
        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

        print("在整个数据集上进行训练和测试...")

        # 决策树分类器，最大深度限制为 7
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)

        # 转换为二值标签
        positive_class = self.rare_types
        binary_labels = np.isin(labels, positive_class).astype(int)

        # 训练和测试均在全集上进行
        clf.fit(data, labels)  # 使用全集训练模型
        y_pred = clf.predict(data)  # 预测全集

        # 转为二值预测标签
        y_pred_binary = np.isin(y_pred, positive_class).astype(int)

        # 计算准确率（原始标签）
        accuracy = accuracy_score(labels, y_pred)

        # 计算 F1、Precision 和 Recall（根据二值标签）
        f1 = f1_score(binary_labels, y_pred_binary, zero_division=0)
        prec = precision_score(binary_labels, y_pred_binary, zero_division=0)
        rec = recall_score(binary_labels, y_pred_binary, zero_division=0)

        # 输出决策树结构
        print("决策树结构：")
        tree_structure = export_text(clf, feature_names=geneNames)
        print(tree_structure)

        # 打印结果
        print(f"准确率: {accuracy:.4f}")
        print(f"F1 分数: {f1:.4f}")
        print(f"Precision 分数: {prec:.4f}")
        print(f"Recall 分数: {rec:.4f}")

        return tree_structure
    
    def parse_tree_structure(self, tree_structure):
        nodes = []
        for line in tree_structure.strip().split("\n"):
            line = line.strip()
            if "class:" not in line:  # 忽略叶子节点
                if "<=" in line:
                    feature, threshold = line.split(" <= ")
                    nodes.append((feature.strip("|- "), float(threshold)))
                elif ">" in line:
                    feature, threshold = line.split(" > ")
                    nodes.append((feature.strip("|- "), float(threshold)))

        seen = set()
        deduplicated_nodes = []
        for node in nodes:
            if node not in seen:
                deduplicated_nodes.append(node)
                seen.add(node)
        return deduplicated_nodes
    


    def sample_cells_equally(self, dataset_name, data, labels, cellNames, k):
        """
        从每个类别中随机选择 k% 的细胞
        """
        unique_labels = np.unique(labels)
        selected_indices = []
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            sample_size = max(1, int(len(indices) * k / 100))  # 确保至少选择一个
            selected_indices.extend(random.sample(list(indices), sample_size))

        indset = {}
        if dataset_name == "Chung":
            indset[10] = [52, 123, 186, 420, 98, 251, 447, 136, 224, 423, 500, 397, 365, 27, 320, 51, 201, 343, 411, 162, 237, 113, 441, 278, 358, 426, 280, 431, 18, 96, 42, 36, 66, 377, 182, 508, 362, 361, 415, 112, 476, 325, 494, 76, 311, 134, 105, 254, 287, 351, 422] 
            indset[20] = [305, 285, 74, 462, 170, 251, 464, 166, 331, 38, 243, 183, 7, 504, 37, 320, 252, 249, 415, 501, 25, 452, 168, 332, 310, 242, 194, 395, 214, 292, 387, 245, 47, 87, 61, 434, 330, 482, 316, 73, 128, 12, 404, 11, 139, 350, 414, 153, 287, 58, 506, 352, 70, 50, 300, 371, 199, 277, 512, 342, 321, 451, 255, 389, 421, 234, 259, 54, 401, 83, 460, 126, 69, 413, 162, 122, 89, 385, 205, 510, 6, 4, 475, 173, 238, 27, 241, 314, 160, 80, 471, 97, 57, 284, 236, 374, 17, 353, 163, 75, 473, 24, 231] 
            indset[30] = [245, 311, 40, 387, 481, 382, 436, 91, 174, 243, 375, 102, 325, 181, 399, 478, 131, 217, 346, 398, 314, 393, 377, 124, 364, 169, 486, 480, 456, 194, 415, 213, 95, 48, 350, 27, 492, 165, 373, 366, 287, 8, 120, 231, 278, 317, 62, 363, 75, 185, 279, 209, 420, 263, 320, 227, 224, 156, 89, 159, 338, 297, 289, 4, 423, 186, 142, 23, 299, 232, 509, 7, 11, 433, 485, 429, 365, 501, 408, 64, 304, 274, 189, 238, 380, 258, 130, 435, 170, 248, 247, 514, 26, 200, 340, 237, 419, 315, 504, 179, 330, 508, 230, 500, 133, 33, 323, 293, 358, 41, 396, 218, 46, 497, 17, 123, 74, 421, 210, 347, 204, 99, 294, 406, 98, 348, 0, 132, 87, 479, 409, 68, 305, 368, 229, 69, 141, 328, 266, 414, 73, 257, 464, 183, 384, 140, 390, 507, 241, 2, 21, 203, 205, 107] 
            indset[40] = [135, 326, 392, 211, 200, 109, 418, 29, 184, 105, 342, 221, 371, 337, 495, 454, 287, 438, 186, 494, 350, 40, 457, 416, 513, 381, 232, 100, 281, 189, 146, 286, 479, 190, 164, 433, 262, 37, 451, 297, 503, 145, 193, 312, 17, 241, 13, 370, 9, 107, 502, 108, 307, 227, 344, 75, 505, 377, 409, 338, 394, 240, 355, 351, 313, 224, 170, 363, 98, 192, 99, 504, 402, 273, 156, 72, 413, 58, 453, 47, 432, 269, 459, 399, 395, 496, 410, 51, 2, 71, 163, 482, 12, 362, 236, 89, 446, 423, 367, 272, 169, 343, 314, 327, 484, 462, 50, 205, 194, 210, 365, 0, 288, 408, 181, 74, 124, 234, 106, 85, 73, 284, 264, 429, 188, 229, 20, 245, 512, 14, 159, 477, 458, 491, 321, 166, 87, 417, 185, 406, 255, 320, 263, 152, 298, 428, 122, 339, 431, 501, 46, 172, 110, 487, 16, 68, 306, 4, 24, 253, 167, 218, 127, 375, 79, 147, 125, 325, 359, 347, 452, 136, 228, 230, 64, 450, 435, 256, 112, 130, 39, 233, 86, 67, 400, 242, 353, 277, 206, 191, 104, 324, 113, 219, 258, 77, 497, 203, 317, 220, 376, 278, 383, 488, 94, 315] 
            indset[50] = [131, 106, 469, 359, 372, 2, 336, 483, 226, 258, 321, 127, 427, 34, 314, 189, 500, 17, 317, 380, 76, 470, 98, 443, 223, 446, 375, 247, 257, 135, 476, 95, 150, 419, 396, 250, 193, 201, 204, 79, 293, 108, 160, 53, 143, 499, 385, 52, 333, 222, 170, 156, 11, 198, 215, 352, 136, 326, 154, 174, 462, 433, 370, 101, 297, 421, 494, 244, 384, 442, 33, 103, 322, 501, 374, 29, 383, 432, 472, 38, 190, 229, 37, 438, 172, 112, 70, 179, 512, 423, 398, 324, 122, 332, 349, 466, 71, 467, 411, 60, 242, 274, 14, 8, 238, 391, 323, 318, 312, 73, 111, 80, 166, 376, 142, 43, 463, 276, 357, 267, 35, 41, 10, 390, 406, 22, 434, 137, 457, 492, 440, 62, 514, 203, 412, 410, 450, 343, 191, 55, 400, 308, 134, 456, 197, 393, 413, 200, 130, 104, 31, 296, 283, 373, 422, 109, 487, 342, 97, 181, 355, 268, 485, 110, 235, 445, 295, 141, 9, 207, 218, 209, 163, 474, 13, 96, 468, 353, 78, 508, 305, 395, 402, 42, 165, 93, 262, 341, 381, 90, 363, 20, 285, 307, 18, 92, 418, 194, 116, 208, 173, 233, 503, 475, 177, 236, 72, 365, 158, 502, 377, 4, 309, 495, 261, 496, 12, 506, 331, 337, 178, 182, 184, 260, 251, 425, 254, 266, 426, 491, 199, 289, 161, 481, 68, 58, 100, 435, 65, 444, 21, 351, 300, 228, 358, 269, 50, 263, 114, 509, 107, 303, 169, 120, 25, 319, 414] 
            indset[60] = [160, 482, 63, 402, 231, 183, 495, 153, 2, 418, 64, 351, 311, 356, 189, 461, 389, 505, 281, 238, 358, 76, 353, 97, 248, 417, 251, 93, 240, 481, 306, 496, 119, 181, 286, 493, 170, 393, 341, 190, 473, 312, 167, 414, 290, 416, 216, 224, 218, 256, 451, 360, 456, 215, 277, 98, 364, 295, 316, 314, 99, 184, 440, 376, 257, 370, 445, 498, 340, 263, 385, 457, 435, 15, 394, 322, 270, 100, 40, 276, 116, 458, 16, 122, 81, 14, 274, 102, 32, 419, 59, 54, 180, 337, 217, 380, 23, 395, 490, 266, 9, 158, 17, 206, 70, 471, 407, 68, 152, 13, 428, 292, 201, 396, 332, 150, 422, 205, 131, 62, 272, 203, 148, 338, 283, 162, 282, 442, 427, 375, 229, 37, 147, 352, 56, 28, 95, 381, 228, 485, 220, 185, 259, 29, 453, 202, 45, 20, 142, 346, 291, 222, 230, 328, 72, 486, 260, 26, 80, 397, 174, 132, 271, 120, 425, 489, 241, 382, 35, 94, 377, 96, 371, 323, 268, 406, 3, 467, 55, 48, 365, 36, 415, 386, 145, 255, 514, 236, 12, 19, 504, 279, 92, 144, 109, 252, 41, 405, 226, 289, 273, 207, 115, 313, 315, 250, 429, 297, 468, 101, 318, 135, 69, 336, 443, 66, 387, 58, 175, 105, 275, 168, 368, 0, 214, 114, 420, 155, 22, 165, 5, 82, 8, 452, 431, 30, 27, 239, 246, 178, 355, 136, 198, 462, 296, 159, 77, 438, 51, 499, 208, 200, 146, 151, 506, 359, 10, 347, 455, 262, 430, 348, 390, 325, 233, 410, 437, 176, 65, 465, 487, 85, 384, 513, 67, 187, 108, 317, 140, 349, 225, 139, 483, 61, 423, 130, 199, 141, 149, 280, 460, 357, 89, 426, 366, 25, 161, 169, 269, 326, 107, 164, 84, 234, 507, 300, 264, 258, 454] 
            indset[70] = [232, 488, 335, 301, 405, 140, 485, 273, 244, 473, 460, 492, 461, 482, 362, 345, 480, 146, 211, 187, 276, 333, 279, 2, 278, 367, 452, 217, 71, 112, 250, 302, 179, 323, 464, 357, 412, 18, 92, 11, 219, 167, 506, 131, 268, 394, 437, 110, 6, 341, 422, 358, 67, 346, 409, 62, 397, 477, 293, 22, 7, 33, 355, 259, 183, 277, 290, 377, 274, 3, 462, 126, 340, 336, 78, 63, 212, 494, 467, 194, 26, 423, 82, 43, 253, 504, 332, 100, 25, 191, 175, 296, 391, 378, 413, 419, 410, 438, 195, 239, 165, 243, 331, 292, 87, 127, 14, 330, 417, 73, 392, 30, 51, 90, 220, 70, 113, 476, 499, 152, 375, 197, 491, 5, 9, 267, 103, 154, 360, 147, 390, 414, 76, 310, 69, 56, 442, 429, 408, 184, 17, 150, 486, 128, 178, 216, 275, 214, 176, 52, 32, 447, 163, 328, 347, 68, 299, 261, 256, 308, 188, 83, 108, 401, 208, 280, 64, 443, 282, 136, 20, 444, 121, 379, 435, 101, 320, 149, 21, 269, 177, 289, 270, 497, 300, 207, 431, 155, 105, 19, 502, 235, 283, 221, 425, 157, 359, 327, 318, 117, 487, 247, 223, 483, 440, 314, 262, 368, 230, 170, 189, 265, 439, 156, 124, 342, 322, 334, 225, 374, 508, 79, 171, 145, 303, 151, 148, 498, 227, 24, 13, 434, 85, 173, 12, 445, 231, 436, 513, 505, 454, 169, 72, 200, 386, 490, 287, 393, 363, 94, 400, 193, 343, 365, 81, 489, 312, 4, 427, 420, 361, 241, 58, 141, 474, 75, 306, 59, 248, 99, 369, 285, 95, 66, 118, 107, 196, 115, 373, 86, 218, 135, 89, 158, 249, 286, 446, 255, 448, 403, 385, 213, 380, 263, 134, 144, 88, 36, 478, 123, 304, 42, 307, 352, 54, 316, 371, 484, 313, 344, 104, 424, 372, 428, 324, 514, 450, 137, 294, 510, 354, 106, 185, 272, 381, 80, 406, 495, 23, 326, 1, 93, 8, 329, 418, 395, 348, 512, 471, 337, 396, 111, 470, 311, 202, 240, 198, 264, 166, 192, 0, 46, 132, 204, 162, 16, 295, 288, 10, 441] 
            indset[80] = [369, 410, 84, 352, 476, 203, 492, 450, 69, 473, 309, 37, 33, 140, 99, 383, 178, 496, 297, 259, 396, 358, 217, 177, 36, 490, 60, 440, 218, 272, 365, 121, 315, 422, 445, 424, 466, 484, 514, 329, 23, 14, 138, 119, 327, 284, 371, 412, 367, 146, 423, 136, 176, 56, 166, 398, 268, 480, 381, 456, 106, 212, 2, 201, 51, 89, 47, 221, 330, 175, 264, 197, 116, 49, 30, 472, 416, 28, 506, 15, 71, 495, 357, 467, 455, 204, 20, 465, 96, 300, 29, 248, 160, 206, 307, 131, 335, 1, 361, 507, 61, 399, 35, 385, 258, 311, 513, 442, 350, 254, 256, 301, 421, 322, 304, 104, 316, 312, 186, 110, 107, 483, 3, 292, 156, 341, 298, 167, 181, 210, 92, 76, 391, 226, 145, 242, 21, 395, 413, 59, 508, 353, 372, 282, 393, 118, 31, 408, 377, 414, 249, 66, 202, 477, 295, 87, 323, 72, 250, 184, 288, 493, 63, 401, 397, 405, 115, 109, 370, 18, 418, 459, 388, 314, 275, 41, 108, 494, 347, 171, 112, 468, 219, 187, 199, 260, 470, 200, 354, 129, 360, 81, 276, 387, 154, 287, 271, 426, 230, 103, 474, 343, 320, 55, 224, 285, 274, 34, 419, 345, 70, 198, 283, 90, 475, 337, 269, 9, 94, 152, 189, 324, 447, 501, 73, 215, 235, 273, 364, 356, 485, 239, 403, 321, 173, 449, 17, 261, 117, 13, 334, 340, 348, 164, 135, 159, 277, 238, 289, 32, 80, 488, 463, 417, 443, 430, 98, 4, 97, 38, 339, 25, 223, 255, 291, 386, 10, 453, 102, 394, 196, 464, 510, 183, 512, 262, 432, 378, 207, 245, 299, 326, 318, 158, 65, 211, 54, 257, 128, 44, 441, 344, 148, 78, 241, 163, 435, 305, 252, 68, 43, 286, 404, 45, 390, 500, 75, 64, 179, 362, 497, 392, 134, 86, 161, 229, 228, 415, 359, 338, 16, 130, 325, 486, 0, 376, 26, 351, 79, 281, 457, 471, 85, 294, 192, 400, 209, 74, 331, 375, 46, 205, 319, 499, 231, 100, 125, 333, 225, 88, 220, 502, 479, 278, 233, 411, 402, 120, 461, 446, 428, 169, 296, 482, 384, 83, 444, 180, 165, 454, 151, 246, 420, 50, 155, 406, 451, 240, 222, 232, 174, 436, 67, 194, 310, 336, 141, 379, 105, 53, 11, 170, 91, 193, 77, 505, 407, 188, 431, 503, 137, 185, 389, 195, 6, 237, 267, 126, 127, 208, 113, 95] 
            indset[90] = [158, 33, 475, 285, 442, 76, 173, 59, 174, 232, 186, 491, 81, 155, 411, 238, 438, 229, 85, 56, 380, 101, 500, 337, 136, 371, 235, 80, 21, 327, 382, 490, 181, 350, 357, 296, 485, 484, 512, 360, 111, 514, 137, 260, 375, 36, 217, 393, 193, 456, 196, 145, 241, 293, 262, 226, 44, 230, 116, 120, 340, 471, 342, 28, 468, 149, 336, 151, 270, 117, 49, 244, 264, 223, 332, 147, 231, 4, 488, 110, 87, 227, 341, 462, 386, 288, 271, 414, 497, 297, 247, 53, 51, 424, 95, 431, 503, 412, 470, 154, 178, 480, 93, 167, 507, 189, 107, 308, 458, 300, 339, 420, 283, 436, 249, 246, 124, 18, 245, 162, 169, 34, 69, 62, 453, 130, 390, 115, 132, 209, 269, 13, 476, 362, 204, 176, 464, 379, 88, 14, 281, 425, 445, 356, 146, 444, 199, 216, 26, 128, 387, 119, 359, 150, 152, 64, 121, 502, 354, 261, 109, 511, 78, 104, 274, 504, 383, 171, 310, 219, 322, 16, 460, 370, 378, 394, 172, 38, 96, 198, 403, 29, 91, 99, 455, 454, 215, 84, 268, 220, 385, 108, 102, 318, 127, 195, 6, 17, 138, 392, 22, 364, 15, 428, 317, 97, 505, 77, 407, 294, 452, 409, 184, 291, 54, 168, 320, 419, 324, 106, 499, 61, 486, 298, 417, 396, 400, 369, 187, 365, 98, 349, 315, 192, 35, 437, 11, 334, 487, 404, 448, 469, 37, 440, 163, 373, 89, 279, 366, 41, 67, 143, 27, 94, 66, 92, 258, 398, 446, 312, 472, 175, 153, 240, 2, 449, 105, 170, 372, 254, 141, 20, 284, 208, 182, 75, 118, 9, 140, 344, 60, 265, 395, 185, 166, 30, 457, 439, 335, 273, 355, 483, 267, 73, 165, 306, 451, 406, 207, 510, 160, 237, 3, 50, 433, 243, 65, 236, 329, 191, 389, 302, 295, 434, 133, 159, 113, 399, 148, 377, 68, 478, 276, 450, 86, 82, 498, 301, 441, 304, 319, 134, 368, 83, 397, 418, 280, 5, 430, 157, 426, 328, 129, 72, 346, 122, 19, 367, 479, 259, 225, 422, 415, 493, 214, 416, 164, 142, 114, 253, 45, 131, 345, 311, 482, 321, 331, 179, 495, 435, 12, 7, 307, 252, 381, 32, 156, 239, 251, 363, 201, 423, 180, 263, 70, 248, 257, 496, 405, 325, 39, 358, 489, 473, 255, 203, 48, 481, 361, 57, 210, 224, 197, 305, 287, 0, 47, 286, 326, 309, 103, 188, 443, 494, 509, 126, 194, 376, 200, 410, 71, 218, 466, 242, 234, 277, 63, 58, 343, 352, 461, 316, 348, 112, 79, 429, 42, 275, 213, 43, 139, 250, 222, 501, 202, 100, 427, 135, 289, 477, 90, 374, 513, 52, 421, 459, 330, 303, 347, 23, 388, 467, 447] 
            selected_indices = indset[k]
        elif dataset_name == "Darmanis":
            indset[10] = [291, 109, 96, 415, 204, 323, 398, 320, 233, 360, 31, 52, 190, 336, 89, 128, 221, 430, 61, 177, 241, 267, 176, 117, 84, 390, 133, 99, 139, 396, 83, 420, 413, 279, 330, 47, 341, 63, 371, 257, 403, 253, 370, 364, 275, 397] 
            indset[20] = [191, 313, 249, 380, 101, 459, 163, 224, 446, 382, 35, 272, 338, 48, 213, 70, 252, 123, 90, 420, 244, 223, 307, 442, 119, 285, 44, 181, 99, 227, 331, 199, 355, 441, 161, 364, 211, 176, 304, 81, 153, 204, 174, 423, 240, 82, 394, 212, 67, 392, 344, 62, 105, 406, 118, 195, 57, 98, 15, 171, 130, 301, 433, 317, 292, 74, 340, 393, 159, 50, 455, 273, 229, 328, 264, 22, 395, 115, 291, 379, 20, 150, 356, 140, 312, 6, 88, 231, 341, 437, 268, 256, 109] 
            indset[30] = [285, 411, 246, 366, 48, 349, 230, 84, 427, 88, 402, 2, 414, 333, 0, 368, 270, 231, 114, 459, 262, 336, 322, 458, 445, 131, 170, 271, 151, 292, 314, 268, 210, 51, 221, 312, 371, 393, 332, 353, 199, 263, 337, 134, 318, 350, 412, 269, 30, 408, 298, 273, 141, 365, 338, 13, 194, 229, 104, 189, 341, 282, 161, 422, 409, 323, 290, 413, 345, 407, 100, 223, 225, 190, 244, 205, 326, 140, 329, 94, 142, 304, 251, 171, 429, 126, 237, 92, 73, 342, 303, 364, 27, 172, 105, 85, 248, 443, 444, 5, 219, 72, 258, 33, 259, 208, 69, 61, 441, 390, 295, 233, 261, 313, 65, 176, 169, 423, 157, 363, 195, 202, 245, 240, 45, 279, 60, 316, 419, 325, 87, 418, 277, 118, 400, 156, 396, 432, 428] 
            indset[40] = [351, 407, 96, 13, 279, 14, 403, 350, 290, 185, 275, 207, 209, 293, 401, 306, 173, 256, 6, 157, 426, 80, 435, 415, 132, 118, 24, 259, 15, 69, 324, 349, 222, 338, 154, 399, 159, 194, 366, 46, 16, 453, 175, 203, 418, 150, 310, 49, 86, 413, 133, 235, 82, 73, 448, 166, 187, 201, 31, 336, 292, 462, 376, 449, 385, 402, 420, 302, 424, 248, 129, 309, 307, 126, 127, 410, 428, 167, 33, 404, 2, 359, 42, 141, 441, 36, 240, 90, 289, 17, 131, 124, 83, 334, 192, 445, 92, 405, 41, 215, 231, 319, 348, 460, 412, 354, 130, 202, 409, 182, 300, 343, 323, 247, 79, 220, 3, 117, 358, 367, 151, 62, 143, 40, 362, 99, 148, 125, 330, 47, 305, 152, 314, 56, 216, 223, 340, 365, 268, 356, 438, 57, 119, 431, 295, 277, 316, 75, 114, 104, 88, 161, 355, 219, 101, 353, 267, 294, 271, 408, 263, 436, 22, 312, 347, 270, 211, 138, 112, 197, 102, 186, 226, 317, 456, 388, 208, 281, 352, 162, 205, 444, 335, 463, 29, 251] 
            indset[50] = [447, 184, 459, 175, 87, 145, 431, 383, 167, 264, 334, 133, 235, 397, 335, 15, 345, 72, 124, 381, 199, 11, 362, 39, 419, 456, 395, 135, 61, 425, 280, 430, 254, 297, 288, 65, 433, 44, 137, 437, 237, 26, 249, 296, 32, 256, 239, 323, 399, 158, 261, 52, 452, 372, 111, 142, 454, 28, 201, 214, 418, 267, 311, 415, 143, 53, 64, 155, 183, 262, 34, 271, 331, 240, 73, 245, 428, 348, 227, 382, 206, 147, 314, 152, 178, 108, 172, 389, 105, 212, 315, 342, 282, 163, 36, 343, 328, 18, 8, 307, 81, 84, 295, 210, 408, 4, 326, 33, 95, 165, 153, 442, 215, 391, 207, 115, 114, 40, 121, 98, 455, 269, 460, 409, 76, 79, 97, 299, 91, 275, 439, 416, 284, 70, 67, 222, 19, 353, 198, 406, 309, 90, 232, 242, 298, 93, 77, 20, 132, 363, 290, 197, 438, 446, 356, 304, 109, 258, 195, 318, 252, 364, 293, 373, 450, 305, 273, 321, 285, 458, 268, 300, 461, 388, 404, 313, 176, 413, 173, 103, 396, 367, 119, 310, 398, 41, 292, 203, 368, 400, 325, 402, 7, 164, 417, 322, 46, 401, 346, 99, 187, 22, 129, 154, 231, 136, 134, 424, 68, 247, 138, 412, 116, 248, 170, 445, 24, 3, 337, 35, 350, 83, 117, 174, 130, 243, 260, 444, 189, 403, 188, 21, 357] 
            indset[60] = [194, 425, 141, 27, 172, 169, 190, 319, 393, 201, 325, 374, 81, 90, 430, 183, 52, 346, 402, 138, 9, 173, 70, 19, 348, 427, 108, 246, 282, 214, 137, 45, 232, 17, 404, 144, 195, 67, 296, 159, 69, 395, 344, 130, 219, 248, 252, 301, 353, 193, 257, 148, 175, 340, 429, 464, 222, 56, 60, 57, 126, 12, 233, 306, 189, 113, 55, 423, 80, 139, 38, 212, 459, 226, 168, 391, 203, 254, 320, 271, 367, 372, 158, 94, 78, 127, 99, 120, 422, 426, 157, 302, 441, 327, 258, 321, 240, 419, 4, 102, 16, 249, 449, 10, 59, 250, 39, 30, 450, 156, 230, 106, 239, 275, 389, 112, 61, 160, 354, 413, 379, 451, 376, 28, 65, 273, 115, 225, 446, 74, 274, 375, 392, 25, 326, 197, 283, 73, 288, 53, 31, 355, 457, 347, 24, 87, 435, 415, 123, 136, 385, 407, 5, 26, 305, 1, 223, 145, 255, 79, 215, 315, 290, 297, 72, 448, 314, 164, 86, 268, 439, 234, 396, 331, 202, 335, 88, 143, 461, 416, 49, 401, 211, 104, 421, 32, 381, 179, 119, 262, 398, 7, 247, 365, 369, 380, 64, 77, 224, 433, 359, 131, 40, 291, 261, 370, 188, 428, 207, 298, 0, 410, 20, 342, 444, 454, 142, 196, 384, 152, 107, 146, 75, 235, 452, 338, 378, 186, 334, 453, 242, 295, 333, 191, 281, 358, 198, 285, 309, 95, 286, 432, 303, 218, 397, 199, 100, 147, 161, 256, 22, 312, 424, 177, 111, 458, 91, 272, 405, 438, 336, 316, 447, 134, 251, 307, 98, 293, 37, 400, 462, 114, 310, 213, 170, 167, 418, 227, 162] 
            indset[70] = [260, 128, 270, 242, 79, 119, 236, 42, 6, 287, 368, 262, 56, 10, 218, 43, 105, 8, 425, 157, 52, 395, 196, 286, 229, 93, 280, 152, 353, 89, 339, 161, 78, 391, 23, 309, 366, 200, 438, 62, 5, 417, 266, 225, 120, 458, 98, 443, 184, 428, 194, 240, 142, 441, 327, 426, 133, 47, 22, 436, 27, 388, 138, 83, 377, 290, 151, 33, 202, 60, 344, 114, 17, 141, 440, 252, 257, 386, 335, 37, 176, 278, 159, 364, 453, 328, 351, 143, 253, 74, 80, 418, 316, 445, 452, 44, 238, 111, 16, 158, 230, 465, 182, 58, 241, 348, 86, 164, 131, 284, 220, 363, 462, 72, 215, 115, 88, 393, 127, 46, 191, 373, 34, 267, 50, 197, 365, 123, 148, 439, 233, 18, 212, 228, 172, 139, 20, 32, 402, 265, 371, 96, 125, 103, 398, 70, 170, 451, 318, 359, 69, 26, 166, 259, 361, 283, 45, 317, 271, 357, 92, 350, 135, 173, 110, 338, 183, 63, 244, 282, 90, 71, 416, 235, 375, 337, 140, 213, 84, 224, 401, 396, 407, 108, 302, 136, 134, 341, 178, 203, 291, 310, 155, 432, 160, 64, 384, 67, 245, 24, 65, 410, 222, 307, 209, 129, 273, 121, 374, 299, 326, 464, 223, 193, 301, 247, 303, 422, 206, 130, 454, 154, 198, 424, 304, 76, 264, 113, 150, 268, 331, 389, 232, 329, 356, 82, 450, 421, 285, 433, 175, 38, 354, 449, 446, 457, 279, 463, 352, 185, 276, 296, 118, 12, 394, 2, 99, 181, 13, 95, 255, 405, 59, 186, 85, 256, 81, 15, 117, 314, 382, 39, 308, 334, 239, 292, 379, 204, 288, 51, 437, 55, 35, 40, 372, 429, 112, 434, 163, 263, 49, 167, 77, 431, 311, 258, 323, 342, 254, 340, 261, 330, 188, 30, 358, 219, 1, 25, 380, 116, 0, 102, 144, 171, 435, 100, 412, 413, 243, 122, 320, 132, 195, 444, 336, 274] 
            indset[80] = [12, 149, 320, 333, 167, 282, 96, 326, 90, 153, 388, 145, 361, 59, 367, 56, 208, 319, 344, 21, 168, 357, 33, 162, 351, 116, 14, 301, 316, 407, 129, 221, 19, 84, 304, 390, 148, 365, 159, 140, 419, 174, 31, 202, 290, 131, 382, 377, 313, 375, 275, 264, 286, 283, 196, 238, 305, 161, 291, 312, 356, 302, 146, 235, 448, 338, 327, 2, 429, 462, 308, 128, 398, 352, 323, 175, 91, 215, 332, 428, 126, 20, 366, 242, 135, 370, 46, 329, 218, 182, 402, 64, 17, 395, 406, 193, 306, 72, 386, 410, 411, 172, 273, 204, 281, 85, 289, 80, 120, 413, 125, 348, 349, 317, 409, 339, 372, 49, 255, 444, 245, 107, 198, 389, 246, 325, 137, 134, 29, 257, 452, 336, 322, 106, 163, 233, 203, 324, 101, 232, 217, 54, 99, 318, 362, 445, 30, 330, 241, 228, 93, 248, 219, 65, 263, 113, 94, 292, 285, 243, 385, 169, 354, 87, 114, 369, 250, 328, 247, 261, 136, 42, 117, 212, 449, 190, 396, 75, 1, 67, 295, 447, 400, 154, 138, 258, 279, 457, 293, 130, 36, 427, 321, 446, 39, 97, 48, 115, 171, 119, 387, 244, 195, 156, 341, 426, 299, 430, 431, 346, 435, 176, 200, 22, 147, 76, 296, 123, 60, 347, 298, 335, 256, 170, 151, 88, 240, 165, 236, 314, 353, 237, 253, 231, 6, 269, 15, 461, 459, 454, 103, 144, 418, 438, 82, 340, 271, 184, 63, 127, 185, 34, 192, 350, 391, 465, 412, 378, 268, 463, 74, 197, 7, 121, 183, 420, 173, 443, 434, 260, 132, 223, 216, 460, 266, 32, 189, 52, 288, 297, 262, 124, 442, 26, 239, 143, 62, 73, 399, 164, 3, 379, 307, 43, 416, 45, 274, 11, 199, 440, 95, 230, 404, 393, 122, 337, 226, 0, 251, 270, 118, 397, 464, 213, 201, 455, 141, 166, 384, 50, 108, 81, 38, 186, 359, 342, 392, 69, 376, 303, 37, 272, 451, 181, 414, 83, 133, 70, 278, 102, 287, 360, 10, 44, 150, 311, 363, 111, 265, 252, 225, 13, 453, 405, 433, 139, 437, 104, 98, 421, 179, 364, 403, 24, 209, 458, 9, 294, 100, 158, 105, 61] 
            indset[90] = [271, 128, 164, 245, 305, 3, 346, 45, 226, 319, 355, 212, 19, 335, 373, 272, 416, 424, 38, 296, 460, 200, 61, 349, 129, 276, 203, 391, 389, 75, 353, 73, 324, 352, 197, 216, 18, 311, 131, 191, 84, 135, 437, 69, 250, 396, 109, 183, 155, 195, 20, 175, 240, 23, 184, 303, 367, 137, 285, 269, 22, 88, 10, 322, 248, 35, 368, 141, 207, 229, 43, 170, 239, 299, 107, 210, 8, 166, 428, 434, 329, 103, 362, 179, 298, 302, 116, 377, 261, 342, 252, 244, 442, 403, 145, 51, 357, 398, 432, 318, 125, 458, 451, 142, 246, 28, 199, 186, 408, 25, 253, 78, 192, 169, 149, 5, 100, 40, 265, 31, 79, 374, 297, 162, 370, 83, 156, 94, 315, 421, 89, 313, 326, 443, 330, 386, 16, 426, 399, 151, 161, 439, 332, 354, 405, 44, 371, 158, 209, 400, 317, 194, 438, 50, 251, 449, 132, 7, 98, 275, 395, 225, 360, 11, 65, 95, 359, 409, 270, 37, 136, 228, 205, 102, 425, 348, 388, 120, 356, 52, 115, 17, 206, 306, 53, 278, 375, 304, 130, 27, 237, 316, 41, 213, 454, 338, 291, 148, 112, 308, 406, 372, 54, 1, 273, 450, 152, 293, 211, 407, 390, 134, 138, 414, 310, 351, 427, 82, 48, 159, 420, 154, 320, 222, 181, 369, 165, 295, 219, 397, 93, 59, 163, 430, 404, 202, 133, 363, 119, 99, 238, 63, 289, 266, 453, 410, 30, 91, 394, 123, 365, 173, 380, 241, 382, 345, 422, 312, 232, 314, 86, 104, 457, 445, 279, 447, 429, 85, 60, 2, 282, 74, 140, 187, 111, 462, 284, 172, 224, 167, 401, 171, 92, 58, 286, 71, 412, 180, 174, 260, 113, 233, 29, 254, 208, 339, 72, 243, 431, 33, 67, 178, 267, 114, 57, 257, 81, 446, 259, 34, 105, 97, 321, 325, 262, 463, 385, 255, 344, 465, 220, 301, 411, 290, 263, 309, 334, 26, 47, 14, 55, 294, 242, 452, 300, 331, 383, 464, 234, 122, 347, 230, 381, 153, 423, 204, 392, 108, 288, 146, 441, 201, 307, 217, 9, 264, 337, 121, 379, 124, 350, 147, 236, 42, 144, 456, 13, 417, 433, 292, 96, 110, 90, 80, 440, 139, 21, 221, 36, 283, 12, 46, 235, 6, 223, 214, 39, 280, 415, 101, 274, 160, 461, 378, 418, 49, 249, 448, 323, 358, 366, 336, 76, 62, 127, 24, 66, 256, 435, 343, 198, 143, 277, 15, 190, 436, 168, 287, 327] 
            selected_indices = indset[k]
        return data[selected_indices], labels[selected_indices], cellNames[selected_indices]


    def scCAD(self, dataset, OOD=False, only_metrics=False, percentage=100):
        print("data preprocessing...")
        self.dataset = dataset
        self.OOD = OOD
        data, labels, geneNames, cellNames = self.preprocess_dataset()
        data, labels, cellNames = self.sample_cells_equally(dataset, data, labels, cellNames, percentage)
        print(f"Sampled {percentage}%")

        #### 数据的基本分析
        value_counts = Counter(labels)  # 统计cell types
        for value, count in value_counts.items():
            print(f"{value}: {count}")

        if self.dataset == "Darmanis":
            self.rare_types = ["endothelial", "opc", "microglia"]
        elif self.dataset == "Chung":
            self.rare_types = ["Stromal"]
        elif self.dataset == "Goolam":
            self.rare_types = ["sixteen", "thirtytwo", "fourEE"]
        elif self.dataset == "Jurkat":
            self.rare_types = ["Jurkat"]
        elif self.dataset == "Marsgt":
            self.rare_types = ["Bcell"]
        elif self.dataset == "Yang":
            self.rare_types = ['Telogen_bulge_stem_cell']

        # If gene and cell names are not provided, scCAD will generate them automatically.
        result, score, sub_clusters, degs_list = scCAD.scCAD(data=data, dataName=dataset, cellNames=cellNames, geneNames=geneNames, save_path=f'./result_scCAD/') 
        '''
        Returned Value :
            result : Rare sub-clusters identified by scCAD: list.
            score : Score of every sub_clusters: np.array[n sub-clusters].
            sub_clusters : Assignment of sub-cluster labels for each cell: np.array[n cells].
            degs_list : List of differentially expressed genes used for rare sub-clusters: list.
        '''

        # If cell names are not provided, please run:
        # cellNames = [str(i) for i in range(data.shape[0])]

        # 合并所有 Counter 统计结果
        rare_counter = Counter()
        for i in result:
            indices = np.where(np.isin(cellNames, i))[0]
            rare_counter.update(labels[indices])
            print(Counter(labels[indices]))
        
        # 计算 OOD 相关统计
        ood_total = sum(rare_counter[key] for key in self.rare_types if key in rare_counter)
        detected_total = sum(rare_counter.values())
        gt_counter = Counter(labels)
        
        # 计算 Precision 和 Recall
        if detected_total == 0:
            precision = 0
        else:
            precision = ood_total / detected_total
        
        if len(self.rare_types) == 0:
            recall = 0
        else:
            recall = ood_total / sum(gt_counter[key] for key in self.rare_types if key in gt_counter)
        
        # 计算 F1 分数
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return f1_score


    def main(self, exp_name, dataset, OOD=False, only_metrics=False):
        print("data preprocessing...")
        self.dataset = dataset
        self.OOD = OOD
        data, labels, geneNames, cellNames = self.preprocess_dataset()

        #### 数据的基本分析
        value_counts = Counter(labels)  # 统计cell types
        for value, count in value_counts.items():
            print(f"{value}: {count}")
        # t-SNE图
        # self.visualize(data, labels)
        # 轮廓系数
        # var = self.rare_cluster_variance(data, labels)
        # print(f"{self.dataset} 轮廓系数: {round(var, 4)}")
        

        if not only_metrics:
            print("select key cells...")
            top_30_gene_indices_per_cell, relative_expression = self.select_key_cells_by_rank(data)
            print("llm reasoning...")
            #### Cross-Query 的推理
            self.llm_reason_mr(labels, geneNames, top_30_gene_indices_per_cell, relative_expression, exp_name)

            #### Gene-Cell Type 语义相关性分析的推理
            # self.llm_reason_correlation(labels, geneNames, top_30_gene_indices_per_cell, relative_expression)
            
            #### 基于决策树的推理
            # tree = self.tree100(data, labels, geneNames, cellNames)
            # tree_nodes = self.parse_tree_structure(tree)
            # self.llm_reason_tree(data, labels, geneNames, tree_nodes, exp_name)
            

        print("compute metrics...")
        #### Cross-Query 的结果
        self.compute_metrics_mr(labels, exp_name)
        self.analysis_mr_confidence(labels, exp_name)

        #### Gene-Cell Type 语义相关性分析的结果
        # self.llm_analyze_correlation(labels, geneNames, top_30_gene_indices_per_cell, relative_expression)
        
        # self.compute_prob(labels, exp_name)

        
def comparsion_plot(y, name):
    x = np.arange(len(y))
    x = x*10 + 10

    # 使用 B 样条插值拟合曲线
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # k=3 表示三次 B 样条曲线
    y_smooth = spl(x_smooth)

    # 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', label='Data Points')  # 原始散点
    plt.plot(x_smooth, y_smooth, color='blue', label='Fitted Curve')  # 拟合曲线
    plt.xlabel("Data Size %")
    plt.ylabel("F1 score")
    plt.title(name)
    plt.legend()
    plt.grid()

    # 显示图形
    plt.savefig(f"F1_{name}.png")

    
if __name__ == "__main__":
    llms = ["gpt-4o-mini", "claude-3-7-sonnet-20250219", "gemini-2.0-Flash", "meta-llama/Meta-Llama-3.1-405B-Instruct", "qwen2.5-72b-instruct"]
    dataset = "Yang"
    task = "ood"
    model_id = 0
    rcd = RareCellDetection(model_name=llms[model_id])
    rcd.main(f"CQC_{dataset}_{task}_{llms[model_id]}", dataset, OOD=True if task == "ood" else False, only_metrics=True)
    # rcd.main(f"CQC_{dataset}_{task}_{llms[model_id]}", dataset, OOD=True if task == "ood" else False, only_metrics=False)
    
    # rcd.scCAD("Darmanis", percentage=100)

    '''
    f1_list = []
    for p in tqdm(range(0, 100, 10)):
        f1s = []
        for i in range(5):
            f1s.append(rcd.scCAD(dataset, percentage=p+10))
        f1_list.append(np.mean(f1s))
    print(f1_list)
    
    
    Darmanis_F1 = [np.float64(0.0), np.float64(0.03076923076923077), np.float64(0.16796536796536796), np.float64(0.06999999999999999), np.float64(0.22287758732081633), np.float64(0.37743076177268164), np.float64(0.4400775947532982), np.float64(0.4501456421988804), np.float64(0.5078651890520873), np.float64(0.4976711461133485)]
    comparsion_plot(np.array(Darmanis_F1), "Darmanis")

    Chung_F1 = [np.float64(0.019999999999999997), np.float64(0.07252747252747252), np.float64(0.07581453634085213), np.float64(0.043115942028985506), np.float64(0.03995484400656814), np.float64(0.06698357249897698), np.float64(0.15413006918563288), np.float64(0.1544427341903098), np.float64(0.15298130374116753), np.float64(0.1799756565013244)]
    comparsion_plot(np.array(Chung_F1), "Chung")
    '''





