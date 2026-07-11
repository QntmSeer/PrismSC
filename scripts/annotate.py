# scripts/annotate.py
import os
import json
import scanpy as sc
import muon as mu
import numpy as np
from log_utils import log_info, log_success, log_warn, log_error

def main():
    sample_h5ad = snakemake.input.h5ad
    output_h5ad = snakemake.output.h5ad
    config = snakemake.config

    model_name = config["params"]["annotation"]["model"]

    os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)

    # Load multimodal data
    try:
        mdata = mu.read_h5mu(sample_h5ad)
    except Exception:
        try:
            mdata = mu.read(sample_h5ad)
        except Exception:
            mdata = mu.MuData({'rna': sc.read_h5ad(sample_h5ad)})

    has_rna = 'rna' in mdata.mod

    if has_rna:
        rna = mdata['rna']
        rna.var_names_make_unique()
        
        # Prepare normalized expression matrix for annotation
        rna_norm = rna.copy()
        sc.pp.normalize_total(rna_norm, target_sum=1e4)
        sc.pp.log1p(rna_norm)
        
        # Run CellTypist
        predictions = None
        try:
            import celltypist
            log_info("PrismSC", f"Loading CellTypist reference model: \033[1;35m{model_name}\033[0m...")
            try:
                model = celltypist.models.Model.load(model_name)
            except Exception:
                celltypist.models.download_models(force=False)
                model = celltypist.models.Model.load(model_name)
                
            log_info("PrismSC", "Running CellTypist cell-type classification...")
            predictions = celltypist.annotate(rna_norm, model=model, majority_voting=True)
        except Exception as e:
            log_warn("PrismSC", f"CellTypist classification failed: {e}. Recovering with marker gene scoring.")

        # Apply predictions or fallback heuristic
        if predictions is not None:
            rna.obs['predicted_labels'] = predictions.predicted_labels['predicted_labels'].values
            if 'majority_voting' in predictions.predicted_labels:
                rna.obs['cell_type'] = predictions.predicted_labels['majority_voting'].values
                rna.obs['cell_type_majority'] = predictions.predicted_labels['majority_voting'].values
            else:
                rna.obs['cell_type'] = predictions.predicted_labels['predicted_labels'].values
        else:
            # Heuristic scoring fallback
            log_info("PrismSC", "Running fallback marker gene scoring...")
            
            # Dynamically determine marker genes based on the selected CellTypist model
            model_lower = model_name.lower()
            if "brain" in model_lower or "cortex" in model_lower:
                markers = {
                    'Neurons': ['MAP2', 'SNAP25', 'SYT1'],
                    'Astrocytes': ['GFAP', 'ALDH1L1', 'AQP4'],
                    'Oligodendrocytes': ['MBP', 'MOG', 'OLIG2'],
                    'Microglia': ['AIF1', 'TMEM119']
                }
            elif "lung" in model_lower:
                markers = {
                    'Epithelial_cells': ['EPCAM', 'KRT5', 'SFTPC'],
                    'Endothelial_cells': ['PECAM1', 'CD34'],
                    'Stromal_cells': ['COL1A1', 'ACTA2'],
                    'Immune_cells': ['PTPRC']
                }
            elif "kidney" in model_lower:
                markers = {
                    'Podocytes': ['NPHS1', 'NPHS2', 'SYNPO'],
                    'Proximal_Tubule_cells': ['LRP2', 'CUBN', 'SLC5A1'],
                    'Loop_of_Henle_cells': ['UMOD', 'SLC12A1'],
                    'Collecting_Duct_cells': ['AQP2', 'FXYD4']
                }
            else:
                # Default to immune panel (Immune_All_Low.pkl, etc.)
                markers = {
                    'T_cells': ['CD3D', 'CD3E', 'CD4', 'CD8A'],
                    'B_cells': ['MS4A1', 'CD19'],
                    'Monocytes': ['CD14', 'LYZ', 'CD68'],
                    'NK_cells': ['GNLY', 'NKG7'],
                    'Granulocytes': ['ELANE', 'MPO']
                }
            
            # Calculate score for each cell type
            for cell_label, genes in markers.items():
                valid_genes = [g for g in genes if g in rna_norm.var_names]
                if len(valid_genes) > 0:
                    sc.tl.score_genes(rna_norm, gene_list=valid_genes, score_name=f"{cell_label}_score")
                else:
                    rna_norm.obs[f"{cell_label}_score"] = 0.0
                    
            # Classify based on highest score
            score_cols = [f"{cl}_score" for cl in markers.keys()]
            scores_df = rna_norm.obs[score_cols]
            max_scores = scores_df.idxmax(axis=1)
            rna.obs['cell_type'] = max_scores.apply(lambda x: x.replace("_score", ""))
            
            # If all scores are zero, classify as Unassigned
            zero_cells = (scores_df.sum(axis=1) == 0)
            rna.obs.loc[zero_cells, 'cell_type'] = 'Unassigned'
            rna.obs['predicted_labels'] = rna.obs['cell_type']
            
        log_info("PrismSC", f"Cell annotation complete. Detected populations: {dict(rna.obs['cell_type'].value_counts())}")

    # Save the annotated MuData object
    mdata.write(output_h5ad)
    log_success("PrismSC", f"Annotated dataset saved successfully to {output_h5ad}.")

if __name__ == "__main__":
    main()
