# scripts/preprocess.py
import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from jsonschema import validate
from log_utils import log_info, log_success, log_warn, log_error

# 1. Deterministic Validation of Cohort Manifest
manifest_path = snakemake.config["cohort_manifest"]
schema_path = "config/cohort_schema.json"

try:
    with open(schema_path, "r") as f:
        schema = json.load(f)
    manifest = pd.read_csv(manifest_path, sep="\t")
    validate(instance={"samples": manifest.to_dict(orient="records")}, schema=schema)
    log_success("PrismSC", "Cohort manifest successfully validated against schema.")
except Exception as e:
    raise ValueError(f"Manifest validation failed: {e}")

# Retrieve configuration parameters dynamically from snakemake
min_genes = snakemake.config["params"]["qc"]["min_genes"]
max_genes = snakemake.config["params"]["qc"]["max_genes"]
pct_mito = snakemake.config["params"]["qc"]["pct_mito"]

input_file = snakemake.input.h5
output_h5ad = snakemake.output.h5ad
output_json = snakemake.output.json

os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# Load data using MuData (or Scanpy if single-modality fallback)
try:
    mdata = mu.read_10x_h5(input_file)
    mdata.var_names_make_unique()
except Exception:
    mdata = mu.MuData({'rna': sc.read_10x_h5(input_file)})
    mdata.var_names_make_unique()

has_rna = 'rna' in mdata.mod
has_atac = 'atac' in mdata.mod

qc_stats = {}

# 2. RNA Preprocessing
if has_rna:
    rna = mdata['rna']
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    keep = (rna.obs.n_genes_by_counts >= min_genes) & \
           (rna.obs.n_genes_by_counts <= max_genes) & \
           (rna.obs.pct_counts_mt <= pct_mito)
    
    qc_stats['rna'] = {
        'n_cells_raw': int(rna.n_obs),
        'n_cells_filtered': int(keep.sum()),
        'median_genes_per_cell': float(rna.obs.loc[keep, 'n_genes_by_counts'].median()),
        'median_counts_per_cell': float(rna.obs.loc[keep, 'total_counts'].median())
    }
    
    # Slice the entire MuData object to keep modality coordinates aligned
    mdata = mdata[keep, :].copy()
    
    # Doublet filtering using Scrublet on RNA modality
    try:
        sc.pp.scrublet(mdata['rna'], verbose=False)
        if 'predicted_doublet' in mdata['rna'].obs:
            keep_singlets = ~mdata['rna'].obs['predicted_doublet']
            mdata = mdata[keep_singlets, :].copy()
            qc_stats['rna']['n_cells_filtered'] = int(mdata.n_obs)
    except Exception as e:
        log_warn("PrismSC", f"Scrublet analysis skipped: {e}")

# 3. ATAC Preprocessing (if present)
if has_atac:
    atac = mdata['atac']
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    qc_stats['atac'] = {
        'n_cells_raw': int(atac.n_obs),
        'n_cells_filtered': int(atac.n_obs),
        'median_peaks_per_cell': float(atac.obs['n_genes_by_counts'].median()),
        'median_counts_per_cell': float(atac.obs['total_counts'].median())
    }

# Save processed dataset (retains multimodal structure)
mdata.write(output_h5ad)

# Write QC json
with open(output_json, "w") as f:
    json.dump(qc_stats, f, indent=4)

log_success("PrismSC", f"Preprocessing complete. QC metrics exported to {output_json}.")
