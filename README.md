# PrismSC: Clinical-Grade Single-Cell Cohort Pipeline

```text
      /\
     /  \   P R I S M   S C
    /____\  [ rna | atac | wnn ]
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Snakemake](https://img.shields.io/badge/Snakemake-Workflow-blue.svg)](https://snakemake.github.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Version](https://img.shields.io/badge/version-v1.0.0--stable-green)](#)

PrismSC is a "one-stop" clinical single-cell cohort processing workflow designed for secure, reproducible, and scalable analysis of joint scRNA-seq and scATAC-seq datasets. It integrates quality control, batch correction, automated cell annotation, lineage fate mapping, and clinical diagnostics HTML reporting.

---

## 🔬 Biological Context & Objectives

PrismSC is designed to answer key clinical questions:
1. **Cellular Composition Analysis**: Automatically identify and quantify cell populations across multiple patient cohorts under different clinical conditions (e.g., Healthy vs. Inflamed vs. Post-Treatment).
2. **Multi-Modal Integration**: Fuse transcriptomic and chromatin accessibility profiles using Weighted Nearest Neighbors (WNN) or probabilistic deep generative models (MultiVI, scVI).
3. **Automated Annotation**: Eliminate manual annotation bias by using pre-trained **CellTypist** immune classifiers.
4. **Developmental Dynamics & Fate Mapping**: Use **CellRank 2** (GPCCA) and **PAGA** to map cell transition probabilities and fate commitments along differentiation trajectories (e.g., Monocyte subset maturation).
5. **Interactive Clinical Reporting**: Consolidate QC statistics, modality projections, cell proportions, and fate mapping drivers into a portable, clinician-ready HTML report.

---

## 🛠️ Pipeline Architecture

```
                       Cohort Manifest (TSV)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Module 1: Preprocessing & Doublet Detection│
        │   • QC Filtering (Mito%, Gene counts)        │
        │   • Scrublet Doublet Removal                 │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Module 2: Cohort Integration (Zarr DB)     │
        │   • scVI (transcriptomics batch correction)  │
        │   • MultiVI / WNN (joint multi-omics)        │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Module 3: Automated Cell-Type Annotation   │
        │   • CellTypist Classifier                    │
        │   • Majority Voting Neighborhood Consensus   │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Module 4: Fate Mapping & Lineage Dynamics  │
        │   • Diffusion Pseudotime & PAGA Trajectories │
        │   • CellRank 2 Fate Absorption Probabilities │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Module 5: Clinician Diagnostic Reporting   │
        │   • Self-contained HTML report with CSS      │
        │   • Base64-embedded high-res vector plots    │
        └──────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
PrismSC/
├── config/
│   ├── config.yaml             # Pipeline configuration params
│   ├── cohort_schema.json       # Manifest JSON validation schema
│   └── cohort_manifest.tsv     # Cohort samples and metadata
├── workflow/
│   ├── Snakefile               # Main Snakemake execution entrypoint
│   └── rules/
│       ├── preprocessing.smk    # Single-sample QC rules
│       ├── integration.smk      # Multi-sample batch correction
│       ├── annotation.smk       # CellTypist cell typing
│       ├── dynamics.smk         # CellRank 2 trajectory inference
│       └── reporting.smk        # Jinja2-based clinical report rule
├── scripts/
│   ├── preprocess.py           # QC and doublet filtering script
│   ├── integrate.py            # Cohort data consolidation and integration
│   ├── annotate.py             # CellTypist classifier script
│   ├── dynamics.py             # Trajectory and plotting script
│   └── generate_clinical_report.py # Jinja2 HTML reporter
├── envs/
│   └── clinical-sc-omics.yaml  # Conda environment dependencies
├── Dockerfile                  # Containerization specification
├── .gitignore                  # Git untracked pattern rules
└── README.md                   # Documentation (this file)
```

---

## 🚀 Getting Started

### Installation

1. Create and activate the conda environment:
   ```bash
   conda env create -f envs/clinical-sc-omics.yaml
   conda activate clinical-sc-omics
   ```

2. Run the Snakemake pipeline:
   ```bash
   snakemake --cores 8
   ```

### Dry Run Verification
To verify the workflow DAG and configuration without executing scripts:
   ```bash
   snakemake -n
   ```

---

## 🐳 Docker Deployment

The pipeline is containerized using a lightweight Docker context (excluding large datasets via `.dockerignore`). 

```bash
# Build the production-ready container image
docker build -t prismsc-pipeline .

# Execute on CPU (mapping results directory)
docker run -v $(pwd)/results:/app/results prismsc-pipeline snakemake --cores 8

# Execute with GPU Acceleration (requires NVIDIA Container Toolkit)
docker run --gpus all -v $(pwd)/results:/app/results prismsc-pipeline snakemake --cores 8
```

---

## ☁️ Cloud & HPC Orchestration

### 1. On-Premise Clusters (SLURM)
PrismSC includes a pre-configured Snakemake execution profile for SLURM cluster submission:
*   **Snakemake v8+ (Plugin Executor):**
    ```bash
    pip install snakemake-executor-plugin-slurm
    snakemake --profile config/slurm
    ```
*   **Snakemake v7 (Legacy fallback):**
    ```bash
    snakemake --cluster "sbatch --partition=standard --cpus-per-task={threads} --mem={resources.mem_mb}" -j 16
    ```

### 2. Cloud Orchestration (AWS Batch)
To run dynamically on AWS serverless compute resources, leverage the official AWS executor plugin to route files to S3 buckets:
```bash
pip install snakemake-executor-plugin-aws-batch snakemake-storage-plugin-s3

snakemake --executor aws-batch \
          --jobs 100 \
          --default-resources s3_bucket=my-single-cell-bucket
```

---

## 🧬 Tissue-Specific Configurations
To run the workflow on organs other than immune cells, modify `params.annotation.model` in [config.yaml](file:///c:/Users/Gebruiker/Documents/Bioinformatics/PrismSC/config/config.yaml). The cell annotation script will automatically download the respective CellTypist neural net and route to matching lineage fallback marker panels if the model fails:

| Organ | CellTypist Model | Fallback Lineage Markers |
| :--- | :--- | :--- |
| **Immune (Default)** | `Immune_All_Low.pkl` | T-cells, B-cells, Monocytes, NK, Granulocytes |
| **Brain** | `Human_Dev_Brain.pkl` | Neurons, Astrocytes, Oligodendrocytes, Microglia |
| **Lung** | `Human_Lung_Atlas.pkl` | Epithelial, Endothelial, Stromal, Immune |
| **Kidney** | `Kidney_Biopsy.pkl` | Podocytes, Proximal Tubule, Loop of Henle, Collecting Duct |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
