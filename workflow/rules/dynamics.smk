# workflow/rules/dynamics.smk
# Developmental fate mapping and cell transition dynamics

rule dynamics:
    input:
        zarr = "results/integrated_cohort.zarr",
        ann_h5ads = expand("results/annotation/{sample}_cell_types.h5ad", sample=samples)
    output:
        umap_rna = "results/plots/umap_rna.png",
        umap_atac = "results/plots/umap_atac.png",
        umap_wnn = "results/plots/umap_wnn.png",
        umap_multivi = "results/plots/umap_multivi.png",
        cellrank_trajectory = "results/plots/cellrank_trajectory.png",
        umap_markers = "results/plots/umap_markers.png",
        trajectory_paga = "results/plots/trajectory_paga.png",
        trajectory_pseudotime = "results/plots/trajectory_pseudotime.png",
        metrics = "results/qc/cohort_integration_metrics.json"
    log:
        "results/logs/dynamics.log"
    script:
        "../../scripts/dynamics.py"
