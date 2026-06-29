# scripts/integrate.py
import os
# Restrict multi-threading to prevent OpenMP/MKL conflicts under WSL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import subprocess
import pandas as pd
import numpy as np
from log_utils import log_info, log_success, log_warn, log_error, print_logo

def main():
    print_logo()
    h5ads = snakemake.input.h5ads
    output_zarr = snakemake.output.zarr
    config = snakemake.config

    # Retrieve parameters
    batch_key = config["params"]["integration"]["batch_key"]
    max_epochs = config["params"]["integration"]["max_epochs"]
    early_stopping = config["params"]["integration"].get("early_stopping", False)
    reference_atlas = config.get("reference_atlas", None)

    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    os.makedirs("results/temp", exist_ok=True)

    temp_rna_path = "results/temp/temp_rna.h5ad"
    temp_atac_path = "results/temp/temp_atac.h5ad"
    python_bin = "/home/qntm/miniforge3/envs/prismsc_gpu/bin/python"

    # 1. Isolated Concatenation
    try:
        log_info("PrismSC", "Launching isolated concatenation process...")
        subprocess.run([
            python_bin, "scripts/integrate_worker.py", "concatenate",
            json.dumps(h5ads), config["cohort_manifest"], temp_rna_path, temp_atac_path
        ], check=True)
    except Exception as e:
        log_error("PrismSC", f"Concatenation failed: {e}")
        sys.exit(1)

    # Auto-detect hardware
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        use_gpu = True
    except Exception:
        use_gpu = False
        
    diagnostics = {
        "requested_method": "multimodal_suite",
        "actual_method": "multimodal_suite",
        "fallback_triggered": False,
        "fallback_reason": "",
        "gpu_used": use_gpu,
        "device_name": "NVIDIA GPU" if use_gpu else "CPU"
    }

    # 2. Isolated SCVI VAE model
    temp_scvi_npy = "results/temp/scvi.npy"
    if os.path.exists(temp_rna_path):
        try:
            log_info("PrismSC", "Launching isolated \033[1;35mscVI VAE\033[0m training subprocess...")
            args = [
                python_bin, "scripts/integrate_worker.py", "scvi",
                temp_rna_path, temp_scvi_npy, batch_key, str(max_epochs), str(use_gpu), str(early_stopping)
            ]
            if reference_atlas:
                args.append(reference_atlas)
            subprocess.run(args, check=True)
        except Exception as e:
            log_warn("PrismSC", f"SCVI failed in subprocess: {e}")

    # 3. Isolated PeakVI VAE model
    temp_peakvi_npy = "results/temp/peakvi.npy"
    if os.path.exists(temp_atac_path):
        try:
            log_info("PrismSC", "Launching isolated \033[1;35mPeakVI VAE\033[0m training subprocess...")
            subprocess.run([
                python_bin, "scripts/integrate_worker.py", "peakvi",
                temp_atac_path, temp_peakvi_npy, batch_key, str(max_epochs), str(use_gpu), str(early_stopping)
            ], check=True)
        except Exception as e:
            log_warn("PrismSC", f"PeakVI failed in subprocess: {e}")

    # 4. Isolated MultiVI VAE model
    temp_multivi_npy = "results/temp/multivi.npy"
    if os.path.exists(temp_rna_path) and os.path.exists(temp_atac_path):
        try:
            log_info("PrismSC", "Launching isolated \033[1;35mMultiVI VAE\033[0m training subprocess...")
            subprocess.run([
                python_bin, "scripts/integrate_worker.py", "multivi",
                temp_rna_path, temp_atac_path, temp_multivi_npy, batch_key, str(max_epochs), str(use_gpu), str(early_stopping)
            ], check=True)
        except Exception as e:
            log_warn("PrismSC", f"MultiVI failed in subprocess: {e}")

    # 5. Isolated WNN integration
    temp_diag_json = "results/temp/diag.json"
    with open(temp_diag_json, "w") as f:
        json.dump(diagnostics, f)
        
    try:
        log_info("PrismSC", "Launching isolated \033[1;35mWNN multimodal integration\033[0m subprocess...")
        subprocess.run([
            python_bin, "scripts/integrate_worker.py", "wnn",
            temp_rna_path, temp_atac_path, output_zarr, temp_scvi_npy, temp_peakvi_npy, temp_multivi_npy, temp_diag_json
        ], check=True)
    except Exception as e:
        log_error("PrismSC", f"WNN failed in subprocess: {e}")
        sys.exit(1)

    # Clean up temporary files
    for temp_f in [temp_rna_path, temp_atac_path, temp_scvi_npy, temp_peakvi_npy, temp_multivi_npy, temp_diag_json]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    log_success("PrismSC", f"Integration process completed successfully. Output saved to {output_zarr}.")
    
    sys.stdout.flush()
    sys.stderr.flush()
    # Force clean exit to avoid WSL segmentation faults in PyTorch runtime destructors
    os._exit(0)

if __name__ == "__main__":
    main()
