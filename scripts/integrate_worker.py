# scripts/integrate_worker.py
import os
# Restrict multi-threading to prevent OpenMP/MKL conflicts under WSL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gc
import json
import pandas as pd
import numpy as np
from log_utils import log_info, log_success, log_warn, log_error, gradient_text, progress_gradient_text

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback

PROGRESS_OFFSETS = {
    "concatenate": 0.0,
    "scvi": 0.2,
    "peakvi": 0.4,
    "multivi": 0.6,
    "wnn": 0.8
}
CURRENT_MODE = "concatenate"

def get_worker_prefix(success_offset=0.0):
    fraction = PROGRESS_OFFSETS.get(CURRENT_MODE, 0.0) + success_offset
    return progress_gradient_text("[Worker]", min(1.0, fraction))

def log_worker_info(message):
    prefix = get_worker_prefix()
    print(f"{prefix} {message}")
    sys.stdout.flush()

def log_worker_success(message):
    prefix = get_worker_prefix(success_offset=0.2)
    print(f"{prefix} \033[1;32m[SUCCESS]\033[0m {message}")
    sys.stdout.flush()

def log_worker_warn(message):
    prefix = get_worker_prefix()
    print(f"{prefix} \033[1;33m[WARN]\033[0m {message}")
    sys.stdout.flush()

class CustomRetroProgressBar(Callback):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.spinner_idx = 0
        self.spinner_chars = ["⡿", "⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟"]
        self.last_loss = 0.0

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        
        loss_val = trainer.callback_metrics.get("train_loss_epoch")
        if loss_val is not None:
            self.last_loss = float(loss_val)
        elif trainer.callback_metrics.get("train_loss_step") is not None:
            self.last_loss = float(trainer.callback_metrics.get("train_loss_step"))
            
        pct = int((epoch / max_epochs) * 100)
        bar_len = 15
        filled_len = int(bar_len * epoch // max_epochs)
        bar = "█" * filled_len + "░" * (bar_len - filled_len)
        
        spinner = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
        self.spinner_idx += 1
        
        start_frac = PROGRESS_OFFSETS.get(self.mode, 0.0)
        fraction = start_frac + 0.2 * (epoch / max_epochs)
        prefix = progress_gradient_text("[Worker]", fraction)
        sys.stdout.write(f"\r{prefix} Progress: [{bar}] {pct}% {spinner} | Epoch {epoch}/{max_epochs} | Loss: {self.last_loss:.2e}")
        sys.stdout.flush()

    def on_train_end(self, trainer, pl_module):
        sys.stdout.write("\n")
        sys.stdout.flush()

def run_scvi(rna_path, out_npy, batch_key, max_epochs, use_gpu, early_stopping, reference_model_path=None):
    import torch
    import scanpy as sc
    import scvi
    
    log_worker_info("Loading RNA dataset...")
    rna = sc.read_h5ad(rna_path)
    
    # Check if a pre-trained reference model path is specified and exists for scArches mapping
    if reference_model_path and os.path.exists(reference_model_path):
        log_worker_info(f"Using reference model at {reference_model_path} for scArches mapping...")
        try:
            model = scvi.model.SCVI.load_query_data(rna, reference_model_path, freeze_dropout=True)
            train_kwargs = {
                "max_epochs": min(max_epochs, 20),
                "enable_progress_bar": False,
                "callbacks": [CustomRetroProgressBar("scvi")]
            }
            if use_gpu == "True":
                train_kwargs["accelerator"] = "gpu"
                train_kwargs["devices"] = 1
                train_kwargs["precision"] = 16
            
            log_worker_info("Training query mapping...")
            model.train(**train_kwargs)
            latent = model.get_latent_representation()
            np.save(out_npy, latent)
            log_worker_success("scArches query mapping complete.")
            return
        except Exception as e:
            log_worker_warn(f"scArches query mapping failed: {e}. Falling back to de novo training.")

    rna_copy = rna.copy()
    sc.pp.normalize_total(rna_copy, target_sum=1e4)
    sc.pp.log1p(rna_copy)
    sc.pp.highly_variable_genes(rna_copy, n_top_genes=2000, batch_key=batch_key)
    hvg_genes = rna_copy.var.highly_variable
    
    rna_filtered = rna[:, hvg_genes].copy()
    scvi.model.SCVI.setup_anndata(rna_filtered, batch_key=batch_key)
    model = scvi.model.SCVI(rna_filtered, n_latent=30)
    
    train_kwargs = {
        "max_epochs": max_epochs,
        "enable_progress_bar": False,
        "callbacks": [CustomRetroProgressBar("scvi")]
    }
    if early_stopping == "True":
        train_kwargs["early_stopping"] = True
    if use_gpu == "True":
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1
        train_kwargs["precision"] = 16
        
    log_worker_info("Training SCVI VAE...")
    model.train(**train_kwargs)
    latent = model.get_latent_representation()
    np.save(out_npy, latent)
    log_worker_success("SCVI de novo training complete.")

def run_peakvi(atac_path, out_npy, batch_key, max_epochs, use_gpu, early_stopping):
    import torch
    import scanpy as sc
    import scvi
    
    log_worker_info("Loading ATAC dataset...")
    atac = sc.read_h5ad(atac_path)
    
    peak_sums = np.array(atac.X.sum(axis=0)).flatten()
    top_indices = np.argsort(peak_sums)[-5000:]
    atac_filtered = atac[:, top_indices].copy()
    
    scvi.model.PEAKVI.setup_anndata(atac_filtered, batch_key=batch_key)
    model = scvi.model.PEAKVI(atac_filtered, n_latent=30)
    
    train_kwargs = {
        "max_epochs": min(max_epochs, 50),
        "enable_progress_bar": False,
        "callbacks": [CustomRetroProgressBar("peakvi")]
    }
    if early_stopping == "True":
        train_kwargs["early_stopping"] = True
    if use_gpu == "True":
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1
        
    log_worker_info("Training PeakVI VAE...")
    model.train(**train_kwargs)
    latent = model.get_latent_representation()
    np.save(out_npy, latent)
    log_worker_success("PeakVI de novo training complete.")

def run_multivi(rna_path, atac_path, out_npy, batch_key, max_epochs, use_gpu, early_stopping):
    import torch
    import scanpy as sc
    import muon as mu
    import scvi
    
    log_worker_info("Loading RNA & ATAC datasets for MultiVI...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    rna_copy = rna.copy()
    sc.pp.normalize_total(rna_copy, target_sum=1e4)
    sc.pp.log1p(rna_copy)
    sc.pp.highly_variable_genes(rna_copy, n_top_genes=2000)
    rna_filtered = rna[:, rna_copy.var.highly_variable].copy()
    
    peak_sums = np.array(atac.X.sum(axis=0)).flatten()
    top_indices = np.argsort(peak_sums)[-5000:]
    atac_filtered = atac[:, top_indices].copy()
    
    mdata_filtered = mu.MuData({
        'rna': rna_filtered,
        'atac': atac_filtered
    })
    
    scvi.model.MULTIVI.setup_mudata(
        mdata_filtered,
        rna_layer=None,
        atac_layer=None,
        modalities={"rna_layer": "rna", "atac_layer": "atac"}
    )
    model = scvi.model.MULTIVI(mdata_filtered)
    
    train_kwargs = {
        "max_epochs": min(max_epochs, 50),
        "enable_progress_bar": False,
        "callbacks": [CustomRetroProgressBar("multivi")]
    }
    if early_stopping == "True":
        train_kwargs["early_stopping"] = True
    if use_gpu == "True":
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1
        
    log_worker_info("Training MultiVI VAE...")
    model.train(**train_kwargs)
    latent = model.get_latent_representation()
    np.save(out_npy, latent)
    log_worker_success("MultiVI de novo training complete.")

def run_concatenate(h5ads_json, cohort_manifest, out_rna, out_atac):
    import scanpy as sc
    import muon as mu
    import anndata as ad
    import scipy.sparse as sp
    
    h5ads = json.loads(h5ads_json)
    log_worker_info(f"Loading {len(h5ads)} sample files for concatenation...")
    datasets = []
    for f in h5ads:
        try:
            datasets.append(mu.read_h5mu(f))
        except Exception:
            try:
                datasets.append(mu.read(f))
            except Exception:
                datasets.append(sc.read_h5ad(f))

    cohort = pd.read_csv(cohort_manifest, sep="\t")

    rnas = []
    atacs = []
    for idx, mdata_obj in enumerate(datasets):
        base = os.path.basename(h5ads[idx])
        sample_id = base.replace("_filtered.h5ad", "")
        
        sample_info = cohort.loc[cohort["sample_id"] == sample_id]
        patient_id = sample_info["patient_id"].values[0] if not sample_info.empty else "unknown"
        condition = sample_info["condition"].values[0] if not sample_info.empty else "unknown"
        
        if 'rna' in mdata_obj.mod:
            rna_mod = mdata_obj.mod['rna'].copy()
            rna_mod.obs['sample_id'] = sample_id
            rna_mod.obs['patient_id'] = patient_id
            rna_mod.obs['condition'] = condition
            rna_mod.obs_names = [f"{sample_id}_{bar}" for bar in rna_mod.obs_names]
            rnas.append(rna_mod)
        if 'atac' in mdata_obj.mod:
            atac_mod = mdata_obj.mod['atac'].copy()
            atac_mod.obs['sample_id'] = sample_id
            atac_mod.obs['patient_id'] = patient_id
            atac_mod.obs['condition'] = condition
            atac_mod.obs_names = [f"{sample_id}_{bar}" for bar in atac_mod.obs_names]
            atacs.append(atac_mod)

    mods = {}
    if len(rnas) > 0:
        adata_rna = ad.concat(rnas, join="inner")
        adata_rna.obs_names_make_unique()
        if sp.issparse(adata_rna.X):
            adata_rna.X.eliminate_zeros()
            adata_rna.X.sort_indices()
            if len(adata_rna.X.indptr) > 0 and adata_rna.X.indptr[0] != 0:
                adata_rna.X.indptr = adata_rna.X.indptr - adata_rna.X.indptr[0]
        mods['rna'] = adata_rna
        
    if len(atacs) > 0:
        adata_atac = ad.concat(atacs, join="inner")
        adata_atac.obs_names_make_unique()
        if sp.issparse(adata_atac.X):
            adata_atac.X.eliminate_zeros()
            adata_atac.X.sort_indices()
            if len(adata_atac.X.indptr) > 0 and adata_atac.X.indptr[0] != 0:
                adata_atac.X.indptr = adata_atac.X.indptr - adata_atac.X.indptr[0]
        mods['atac'] = adata_atac

    if 'rna' in mods and 'atac' in mods:
        common_cells = mods['rna'].obs_names.intersection(mods['atac'].obs_names)
        log_worker_info(f"Filtering cohort to {len(common_cells)} common cells...")
        mods['rna'] = mods['rna'][common_cells].copy()
        mods['atac'] = mods['atac'][common_cells].copy()

    if 'rna' in mods:
        mods['rna'].write_h5ad(out_rna)
    if 'atac' in mods:
        mods['atac'].write_h5ad(out_atac)
    log_worker_success("Concatenation complete.")

def sanitize_sparse_matrices(mdata):
    import scipy.sparse as sp
    
    def sanitize_matrix(mat, name):
        if sp.issparse(mat):
            if hasattr(mat, "indptr") and len(mat.indptr) > 0:
                if mat.indptr[0] != 0:
                    start_val = mat.indptr[0]
                    log_worker_info(f"Sanitizing sparse matrix {name}: shifting indptr starting at {start_val} to 0")
                    mat.indptr = mat.indptr - start_val
            mat.eliminate_zeros()
            mat.sort_indices()
            
    # Sanitize root obsp/varp
    for k in list(mdata.obsp.keys()):
        sanitize_matrix(mdata.obsp[k], f"mdata.obsp['{k}']")
    for k in list(mdata.varp.keys()):
        sanitize_matrix(mdata.varp[k], f"mdata.varp['{k}']")
        
    # Sanitize each modality
    for mod_name in mdata.mod.keys():
        mod = mdata.mod[mod_name]
        sanitize_matrix(mod.X, f"mod['{mod_name}'].X")
        for k in list(mod.obsm.keys()):
            sanitize_matrix(mod.obsm[k], f"mod['{mod_name}'].obsm['{k}']")
        for k in list(mod.obsp.keys()):
            sanitize_matrix(mod.obsp[k], f"mod['{mod_name}'].obsp['{k}']")
        for k in list(mod.varp.keys()):
            sanitize_matrix(mod.varp[k], f"mod['{mod_name}'].varp['{k}']")

def run_wnn(rna_path, atac_path, out_zarr, scvi_npy, peakvi_npy, multivi_npy, diagnostics_json_path):
    import scanpy as sc
    import muon as mu
    import scipy.sparse as sp
    
    mods = {}
    if os.path.exists(rna_path):
        mods['rna'] = sc.read_h5ad(rna_path)
    if os.path.exists(atac_path):
        mods['atac'] = sc.read_h5ad(atac_path)
        
    mdata = mu.MuData(mods)
    
    # 1. Sanitize raw loaded mod matrices BEFORE scanpy pca/neighbors computations
    sanitize_sparse_matrices(mdata)
    
    if os.path.exists(scvi_npy):
        latent = np.load(scvi_npy)
        mdata.obsm['X_scvi'] = latent
        mdata['rna'].obsm['X_scvi'] = latent
    if os.path.exists(peakvi_npy):
        latent = np.load(peakvi_npy)
        mdata.obsm['X_peakvi'] = latent
        mdata['atac'].obsm['X_peakvi'] = latent
    if os.path.exists(multivi_npy):
        latent = np.load(multivi_npy)
        mdata.obsm['X_multivi'] = latent
        
    log_worker_info("Computing WNN integration...")
    try:
        rna = mdata.mod['rna']
        if 'X_pca' not in rna.obsm:
            rna_copy = rna.copy()
            sc.pp.normalize_total(rna_copy, target_sum=1e4)
            sc.pp.log1p(rna_copy)
            sc.pp.highly_variable_genes(rna_copy, n_top_genes=2000)
            rna_filtered_pca = rna_copy[:, rna_copy.var.highly_variable].copy()
            sc.pp.pca(rna_filtered_pca)
            rna.obsm['X_pca'] = rna_filtered_pca.obsm['X_pca']
        sc.pp.neighbors(rna, use_rep='X_pca', n_neighbors=15)
        
        atac = mdata.mod['atac']
        if 'X_lsi' not in atac.obsm:
            mu.atac.pp.tfidf(atac, scale_factor=1e4)
            mu.atac.tl.lsi(atac)
        atac.obsm['X_lsi_clean'] = atac.obsm['X_lsi'][:, 1:]
        sc.pp.neighbors(atac, use_rep='X_lsi_clean', n_neighbors=15)
        
        # 2. Sanitize again after neighbors calculations to clean scanpy obsp matrices
        sanitize_sparse_matrices(mdata)
        
        # Namespace-scoped monkey-patch to bypass muon/scipy index pointer starts with 0 bug
        import muon._core.preproc as preproc
        orig_csr = preproc.csr_matrix
        
        def patched_csr(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], tuple) and len(args[0]) == 3:
                data, indices, indptr = args[0]
                if len(indptr) > 1:
                    # Fix non-monotonic indptr (muon candidate graph creation bug with approx neighbors)
                    diffs = np.diff(indptr)
                    if np.any(diffs < 0):
                        n_rows = len(indptr) - 1
                        n_elements_per_row = len(indices) // n_rows
                        indptr = np.arange(n_rows + 1, dtype=indptr.dtype) * n_elements_per_row
                    # Shift indptr to start with 0
                    if indptr[0] != 0:
                        indptr = indptr - indptr[0]
                args = ((data, indices, indptr),) + args[1:]
            return orig_csr(*args, **kwargs)
            
        preproc.csr_matrix = patched_csr
        
        try:
            mu.pp.neighbors(mdata)
        finally:
            preproc.csr_matrix = orig_csr
            
        mu.tl.umap(mdata)
        mdata.obsm['X_wnn_umap'] = mdata.obsm['X_umap']
        log_worker_success("WNN integration and UMAP complete.")
    except Exception as e:
        log_worker_warn(f"WNN calculation failed: {e}")
        
    if os.path.exists(diagnostics_json_path):
        with open(diagnostics_json_path, "r") as f:
            diagnostics = json.load(f)
        mdata.uns["integration_diagnostics"] = json.dumps(diagnostics)
        
    mdata.write_zarr(out_zarr)
    log_worker_success(f"Integrated cohort saved to {out_zarr}.")

if __name__ == "__main__":
    CURRENT_MODE = sys.argv[1]
    mode = CURRENT_MODE
    
    if mode == "scvi":
        ref_model = sys.argv[8] if len(sys.argv) > 8 else None
        run_scvi(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7], ref_model)
    elif mode == "peakvi":
        run_peakvi(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7])
    elif mode == "multivi":
        run_multivi(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7], sys.argv[8])
    elif mode == "concatenate":
        run_concatenate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "wnn":
        run_wnn(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
