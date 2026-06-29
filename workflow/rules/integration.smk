# workflow/rules/integration.smk
# Cohort batch correction and atlas reference mapping

rule integrate:
    input:
        h5ads = expand("results/preprocessing/{sample}_filtered.h5ad", sample=samples)
    output:
        zarr = directory("results/integrated_cohort.zarr")
    log:
        "results/logs/integrate.log"
    script:
        "../../scripts/integrate.py"
