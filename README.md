# How Effective is In-Context Learning with Large Language Models for Rare Cell Identification in Single-Cell Expression Data?

## Overview
The recent development of single-cell genomics requires more powerful computational tools to differentiate between different phenotypes. Rare cell identification has been one of the most important challenges in this area. Traditional data-driven approaches typically rely on feature selection techniques to identify key genes for anomaly detection, often requiring extensive training data or domain-specific knowledge.

In contrast, large language models (LLMs) have demonstrated strong generalization abilities in various scientific research fields, presenting new opportunities for rare cell identification. This repository accompanies our paper, where we conduct the first comprehensive evaluation of in-context learning with LLMs for rare cell identification. Our approach employs a chain-of-thought prompting strategy, integrating latent space analysis and cross-query comparisons to generate scores for identifying rare cells.

## Key Contributions
- **First evaluation of LLMs for rare cell identification** using in-context learning.
- **Novel prompting strategy** combining chain-of-thought reasoning with latent space analysis and cross-query comparisons.
- **Competitive performance** of LLMs compared with traditional optimization-based methods on benchmark datasets.
- **Minimal dependence on extensive training data** or expert-defined feature selection, demonstrating the generalization potential of LLMs in genomics.

## Repository Structure
```
├── data/                  # Benchmark datasets for rare cell identification
├── src/                   # Implementation of our methodology
│   ├── preprocessing.py   # Data preprocessing scripts
│   ├── llm_prompting.py   # Chain-of-thought prompting strategy
│   ├── evaluation.py      # Performance evaluation scripts
├── results/               # Experimental results and analysis
├── README.md              # Project documentation
└── requirements.txt       # Required dependencies
```

## Installation
To set up the environment, clone this repository and install the required dependencies:
```sh
$ git clone https://github.com/RareCellAgent.git
$ cd RareCellAgent
$ pip install -r requirements.txt
```

## Usage
### 1. Data Preprocessing
Prepare the single-cell expression datasets and apply preprocessing:
```sh
$ python src/preprocessing.py --input data/raw_data.csv --output data/processed_data.csv
```

### 2. Running LLM-Based Rare Cell Identification
Execute the LLM-based rare cell identification pipeline:
```sh
$ python src/llm_prompting.py --input data/processed_data.csv --output results/llm_predictions.csv
```

### 3. Evaluating Model Performance
Assess the performance of the LLM-based approach against traditional methods:
```sh
$ python src/evaluation.py --predictions results/llm_predictions.csv --ground_truth data/labels.csv
```

## Benchmark Datasets
We evaluate our approach on publicly available single-cell expression datasets, including:
- **Chung**
- **Darmanis**
- **Goolam**
- **Immuno**

## Citation

If you find this repository useful, please cite our paper:

```
@article{yao2025rarecellagent,
  title={How Effective is In-Context Learning with Large Language Models for Rare Cell Identification in Single-Cell Expression Data?},
  author={Yao, Huaiyuan, Zhenxiao Cao and Xiao Luo etc.},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```
