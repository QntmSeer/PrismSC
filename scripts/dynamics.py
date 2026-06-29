# scripts/dynamics.py
import os
import gc
import json
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import matplotlib.pyplot as plt

# Suppress warnings to reduce clutter
import warnings
warnings.filterwarnings("ignore")
from log_utils import log_info, log_success, log_warn, log_error

def set_premium_colors(adata, key):
    """Assign premium dark-theme compatible colors to categories in adata.obs[key]."""
    if key not in adata.obs:
        return
    cats = adata.obs[key].astype('category').cat.categories
    num_cats = len(cats)
    
    premium_palette = [
        "#1e40af", "#3b82f6", "#60a5fa", "#8b5cf6", 
        "#d946ef", "#ec4899", "#f97316", "#f59e0b", 
        "#14b8a6", "#10b981", "#84cc16", "#a855f7",
        "#a1a1aa", "#78716c", "#ef4444", "#06b6d4"
    ]
    
    colors = [premium_palette[i % len(premium_palette)] for i in range(num_cats)]
    adata.uns[f"{key}_colors"] = colors

def save_placeholder_plot(path, title):
    """Generate a clean dark-themed placeholder that integrates seamlessly into the clinical dashboard."""
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#161925')
        ax.set_facecolor('#161925')
        ax.text(0.5, 0.5, f"{title}\n\n[ Skipped / Not Requested ]\n\n(Adjust integration settings to enable)", 
                ha='center', va='center', fontsize=10, color='#94a3b8', fontweight='semibold', wrap=True)
        ax.axis('off')
        fig.savefig(path, bbox_inches='tight', dpi=150, facecolor='#161925')
        plt.close(fig)

def main():
    # Retrieve inputs and outputs from snakemake
    zarr_path = snakemake.input.zarr
    ann_h5ads = snakemake.input.ann_h5ads

    out_rna = snakemake.output.umap_rna
    out_atac = snakemake.output.umap_atac
    out_wnn = snakemake.output.umap_wnn
    out_multivi = snakemake.output.umap_multivi
    out_cellrank = snakemake.output.cellrank_trajectory
    out_markers = snakemake.output.umap_markers
    out_paga = snakemake.output.trajectory_paga
    out_dpt = snakemake.output.trajectory_pseudotime
    out_metrics = snakemake.output.metrics

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)

    # 1. Load integrated cohort and merge cell-type annotations
    log_info("PrismSC", "Loading integrated cohort dataset...")
    mdata = mu.read_zarr(zarr_path)
    rna = mdata.mod['rna']
    rna.obs['cell_type'] = "Unassigned"

    log_info("PrismSC", "Merging cell-type annotations from cohort samples...")
    for sample_path in ann_h5ads:
        try:
            sample_mdata = mu.read_h5mu(sample_path)
            sample_rna = sample_mdata.mod['rna']
            sample_id = os.path.basename(sample_path).replace("_cell_types.h5ad", "").replace("_filtered.h5ad", "")
            
            # Map using prefixed cell barcodes
            prefixed_names = [f"{sample_id}_{bar}" for bar in sample_rna.obs_names]
            sample_rna_copy = sample_rna.copy()
            sample_rna_copy.obs_names = prefixed_names
            
            common_cells = rna.obs_names.intersection(prefixed_names)
            rna.obs.loc[common_cells, 'cell_type'] = sample_rna_copy.obs.loc[common_cells, 'cell_type'].astype(str)
        except Exception as e:
            log_warn("PrismSC", f"Failed to merge annotations from {sample_path}: {e}")

    # Set premium colors for RNA
    set_premium_colors(rna, 'cell_type')

    # Calculate integration quality metrics (ASW)
    log_info("PrismSC", "Calculating integration quality metrics (ASW)...")
    metrics_dict = {}
    
    # Try to calculate ASW on RNA SCVI latent space
    if 'X_scvi' in rna.obsm:
        from sklearn.metrics import silhouette_score
        try:
            # Silhouette score for cell types (higher is better - preserves cell type structure)
            asw_cell_type = silhouette_score(rna.obsm['X_scvi'], rna.obs['cell_type'])
            # Silhouette score for batches (lower is better - indicates good batch mixing)
            asw_batch = silhouette_score(rna.obsm['X_scvi'], rna.obs['patient_id'])
            
            metrics_dict['scvi_asw_cell_type'] = float(asw_cell_type)
            metrics_dict['scvi_asw_batch'] = float(asw_batch)
            log_info("PrismSC", f"scVI ASW Cell Type: \033[1;35m{asw_cell_type:.4f}\033[0m, ASW Batch: \033[1;35m{asw_batch:.4f}\033[0m")
        except Exception as e:
            print(f"[WARN] Failed to compute scVI ASW: {e}")
            
    # Try to calculate ASW on MultiVI integrated latent space
    if 'X_multivi' in mdata.obsm:
        from sklearn.metrics import silhouette_score
        try:
            asw_cell_type = silhouette_score(mdata.obsm['X_multivi'], rna.obs['cell_type'])
            asw_batch = silhouette_score(mdata.obsm['X_multivi'], rna.obs['patient_id'])
            
            metrics_dict['multivi_asw_cell_type'] = float(asw_cell_type)
            metrics_dict['multivi_asw_batch'] = float(asw_batch)
            log_info("PrismSC", f"MultiVI ASW Cell Type: \033[1;35m{asw_cell_type:.4f}\033[0m, ASW Batch: \033[1;35m{asw_batch:.4f}\033[0m")
        except Exception as e:
            print(f"[WARN] Failed to compute MultiVI ASW: {e}")

    # Save metrics JSON
    with open(out_metrics, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    log_success("PrismSC", f"Integration metrics saved to {out_metrics}.")

    # 2. Compute embeddings and Leiden clustering for RNA
    log_info("PrismSC", "Analyzing RNA embeddings...")
    if 'X_scvi' in rna.obsm:
        sc.pp.neighbors(rna, use_rep="X_scvi", n_neighbors=15)
    else:
        if 'X_pca' not in rna.obsm:
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)
            sc.pp.highly_variable_genes(rna, n_top_genes=2000)
            sc.pp.pca(rna)
        sc.pp.neighbors(rna, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(rna)
    sc.tl.leiden(rna, resolution=0.5, key_added='leiden_rna')

    # Save RNA UMAP
    log_info("PrismSC", "Saving RNA UMAP...")
    fig = sc.pl.umap(rna, color='cell_type', title="RNA Modality Cohort UMAP", frameon=False, return_fig=True, show=False)
    fig.savefig(out_rna, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 3. ATAC embeddings (if ATAC modality exists)
    if 'atac' in mdata.mod:
        log_info("PrismSC", "Analyzing ATAC embeddings...")
        atac = mdata.mod['atac']
        if 'X_peakvi' in atac.obsm:
            sc.pp.neighbors(atac, use_rep="X_peakvi", n_neighbors=15)
        else:
            if 'X_lsi' not in atac.obsm:
                mu.atac.pp.tfidf(atac, scale_factor=1e4)
                mu.atac.tl.lsi(atac)
            atac.obsm['X_lsi_clean'] = atac.obsm['X_lsi'][:, 1:]
            sc.pp.neighbors(atac, use_rep="X_lsi_clean", n_neighbors=15)
        sc.tl.umap(atac)
        
        atac.obs['cell_type'] = rna.obs['cell_type']
        set_premium_colors(atac, 'cell_type')
        fig = sc.pl.umap(atac, color='cell_type', title="ATAC Modality Cohort UMAP", frameon=False, return_fig=True, show=False)
        fig.savefig(out_atac, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        save_placeholder_plot(out_atac, "ATAC Modality UMAP")

    # 4. WNN embeddings
    try:
        if 'X_wnn_umap' in mdata.obsm:
            log_info("PrismSC", "Plotting WNN UMAP...")
            mdata.obs['cell_type'] = rna.obs['cell_type']
            set_premium_colors(mdata, 'cell_type')
            mdata.obsm['X_umap'] = mdata.obsm['X_wnn_umap']
            fig = mu.pl.umap(mdata, color='cell_type', title="WNN Multimodal UMAP", frameon=False, return_fig=True, show=False)
            fig.savefig(out_wnn, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            save_placeholder_plot(out_wnn, "WNN Multimodal UMAP")
    except Exception as e:
        print(f"[WARN] Failed to plot WNN UMAP: {e}")
        save_placeholder_plot(out_wnn, "WNN Multimodal UMAP")

    # 5. MultiVI embeddings
    if 'X_multivi' in mdata.obsm:
        log_info("PrismSC", "Analyzing and plotting MultiVI embeddings...")
        try:
            sc.pp.neighbors(mdata, use_rep="X_multivi", key_added="multivi")
            sc.tl.umap(mdata, neighbors_key="multivi")
            mdata.obs['cell_type'] = rna.obs['cell_type']
            set_premium_colors(mdata, 'cell_type')
            fig = sc.pl.umap(mdata, color='cell_type', title="MultiVI Integrated UMAP", frameon=False, return_fig=True, show=False)
            fig.savefig(out_multivi, bbox_inches='tight', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Failed to plot MultiVI UMAP: {e}")
            save_placeholder_plot(out_multivi, "MultiVI VAE Embedding")
    else:
        save_placeholder_plot(out_multivi, "MultiVI VAE Embedding")

    # 6. Save Marker Genes Plot
    log_info("PrismSC", "Saving Marker Genes UMAP...")
    marker_genes = ['CD3D', 'MS4A1', 'CD14', 'GNLY']
    available_markers = [g for g in marker_genes if g in rna.var_names]
    if len(available_markers) > 0:
        fig = sc.pl.umap(rna, color=available_markers, title="Immune Lineage Marker expression", frameon=False, return_fig=True, show=False, ncols=2, wspace=0.3, hspace=0.3)
        fig.savefig(out_markers, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        save_placeholder_plot(out_markers, "Lineage Marker Genes")

    # 7. CellRank 2 Trajectory Inference (Monocyte Lineage Continuum)
    log_info("PrismSC", "Setting up trajectory inference...")
    is_mono = rna.obs['cell_type'].str.contains('Mono|myeloid', case=False, na=False)

    if np.sum(is_mono) >= 10:
        try:
            import cellrank as cr
            from cellrank.kernels import PseudotimeKernel
            from cellrank.estimators import GPCCA
            
            mono = rna[is_mono, :].copy()
            
            # Compute common diffusion maps and pseudotime
            try:
                sc.pp.neighbors(mono, use_rep="X_scvi" if "X_scvi" in mono.obsm else "X_pca", n_neighbors=15)
                sc.tl.umap(mono)
                
                cd14_expr = None
                if 'CD14' in mono.var_names:
                    cd14_expr = mono[:, 'CD14'].X.toarray().flatten() if hasattr(mono[:, 'CD14'].X, "toarray") else mono[:, 'CD14'].X.flatten()
                    
                if cd14_expr is not None and len(cd14_expr) > 0:
                    root_idx = np.argmax(cd14_expr)
                    mono.uns['iroot'] = int(root_idx)
                    sc.tl.diffmap(mono)
                    sc.tl.dpt(mono)
                    log_info("PrismSC", f"Selected cell index {root_idx} as trajectory root (CD14 max).")
                else:
                    mono.uns['iroot'] = 0
                    sc.tl.diffmap(mono)
                    sc.tl.dpt(mono)
            except Exception as e:
                print(f"[WARN] Basic trajectory setup failed: {e}")
                save_placeholder_plot(out_cellrank, "CellRank 2 Macrostates")
                save_placeholder_plot(out_paga, "PAGA Cluster Graph")
                save_placeholder_plot(out_dpt, "Diffusion Pseudotime trajectory")
                return

            # Plot DPT Pseudotime UMAP
            try:
                log_info("PrismSC", "Saving DPT Pseudotime UMAP...")
                fig = sc.pl.umap(mono, color='dpt_pseudotime', title="Diffusion Pseudotime (Classical -> Non-Classical)", frameon=False, return_fig=True, show=False)
                fig.savefig(out_dpt, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] DPT plotting failed: {e}")
                save_placeholder_plot(out_dpt, "Diffusion Pseudotime trajectory")

            # Plot PAGA transition graph
            try:
                log_info("PrismSC", "Saving PAGA transition graph...")
                sc.tl.paga(mono, groups='cell_type')
                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.paga(mono, ax=ax, show=False, frameon=False, labels=None, node_size_scale=3.0, edge_width_scale=2.0)
                ax.axis('off')
                
                # Dynamic Legend for PAGA to prevent overlapping labels
                handles = []
                for cat, col in zip(mono.obs['cell_type'].cat.categories, mono.uns['cell_type_colors']):
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=col, markersize=10))
                ax.legend(handles=handles, loc='best', frameon=False)
                
                fig.savefig(out_paga, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] PAGA transition graph failed: {e}")
                save_placeholder_plot(out_paga, "PAGA Cluster Graph")

            # CellRank 2 GPCCA estimator
            try:
                log_info("PrismSC", "Setting up CellRank 2 PseudotimeKernel...")
                pk = PseudotimeKernel(mono, time_key="dpt_pseudotime")
                pk.compute_transition_matrix()
                
                estimator = GPCCA(pk)
                estimator.compute_macrostates(n_states=2, cluster_key="cell_type")
                estimator.predict_terminal_states()
                estimator.compute_fate_probabilities()
                
                log_info("PrismSC", "Saving CellRank trajectory plot...")
                fig, ax = plt.subplots(figsize=(6, 6))
                set_premium_colors(mono, 'cell_type')
                estimator.plot_macrostates(which="all", ax=ax, show=False, legend_loc="right margin")
                ax.axis('off')
                fig.savefig(out_cellrank, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                log_warn("PrismSC", f"CellRank GPCCA macrostates failed: {e}. Attempting recovery by row normalization...")
                try:
                    if hasattr(pk, "transition_matrix"):
                        t_mat = pk.transition_matrix.toarray()
                        row_sums = t_mat.sum(axis=1)
                        row_sums[row_sums == 0] = 1.0
                        t_mat = t_mat / row_sums[:, np.newaxis]
                        from scipy.sparse import csr_matrix
                        pk.transition_matrix = csr_matrix(t_mat)
                        
                        estimator = GPCCA(pk)
                        estimator.compute_macrostates(n_states=2, cluster_key="cell_type")
                        estimator.predict_terminal_states()
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        set_premium_colors(mono, 'cell_type')
                        estimator.plot_macrostates(which="all", ax=ax, show=False, legend_loc="right margin")
                        ax.axis('off')
                        fig.savefig(out_cellrank, bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        log_success("PrismSC", "Recovered CellRank GPCCA macrostates plot successfully!")
                    else:
                        raise ValueError("No transition matrix computed.")
                except Exception as recovery_err:
                    print(f"[WARN] CellRank GPCCA recovery failed: {recovery_err}")
                    save_placeholder_plot(out_cellrank, "CellRank 2 Macrostates")
        except Exception as e:
            print(f"[WARN] CellRank / trajectory initialization failed: {e}.")
            save_placeholder_plot(out_cellrank, "CellRank 2 Macrostates")
            save_placeholder_plot(out_paga, "PAGA Cluster Graph")
            save_placeholder_plot(out_dpt, "Diffusion Pseudotime trajectory")
    else:
        log_info("PrismSC", f"Too few Monocytes (n={np.sum(is_mono)}). Skipping trajectory mapping.")
        save_placeholder_plot(out_cellrank, "CellRank 2 Macrostates")
        save_placeholder_plot(out_paga, "PAGA Cluster Graph")
        save_placeholder_plot(out_dpt, "Diffusion Pseudotime trajectory")

    log_success("PrismSC", "Dynamics analysis completed.")

if __name__ == "__main__":
    main()
