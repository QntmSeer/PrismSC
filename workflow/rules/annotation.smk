# workflow/rules/annotation.smk
# Automated cell-type annotation using CellTypist classifiers

rule annotate:
    input:
        h5ad = "results/preprocessing/{sample}_filtered.h5ad",
        zarr = "results/integrated_cohort.zarr"
    output:
        h5ad = "results/annotation/{sample}_cell_types.h5ad"
    log:
        "results/logs/annotate/{sample}.log"
    script:
        "../../scripts/annotate.py"
