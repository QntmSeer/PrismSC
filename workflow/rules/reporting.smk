# workflow/rules/reporting.smk
# Clinician-facing diagnostics HTML report generation

rule generate_report:
    input:
        qcs = expand("results/qc/{sample}_qc_metrics.json", sample=samples),
        plots = [
            "results/plots/umap_rna.png",
            "results/plots/umap_atac.png",
            "results/plots/umap_wnn.png",
            "results/plots/umap_multivi.png",
            "results/plots/cellrank_trajectory.png",
            "results/plots/umap_markers.png",
            "results/plots/trajectory_paga.png",
            "results/plots/trajectory_pseudotime.png"
        ],
        zarr = "results/integrated_cohort.zarr",
        metrics = "results/qc/cohort_integration_metrics.json",
        summary = "results/qc/cohort_qc_summary.json",
        annotations = expand("results/annotation/{sample}_cell_types.h5ad", sample=samples)
    output:
        report = "results/reports/cohort_clinical_report.html"
    log:
        "results/logs/generate_report.log"
    script:
        "../../scripts/generate_clinical_report.py"
