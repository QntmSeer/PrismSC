# workflow/rules/preprocessing.smk
# Quality control and doublet filtering rules

rule preprocess:
    input:
        h5 = lambda wildcards: cohort.loc[cohort["sample_id"] == wildcards.sample, "file_path"].values[0]
    output:
        h5ad = "results/preprocessing/{sample}_filtered.h5ad",
        json = "results/qc/{sample}_qc_metrics.json"
    log:
        "results/logs/preprocess/{sample}.log"
    script:
        "../../scripts/preprocess.py"

rule aggregate_qc_rust:
    input:
        qcs = expand("results/qc/{sample}_qc_metrics.json", sample=samples),
        manifest = config["cohort_manifest"]
    output:
        summary = "results/qc/cohort_qc_summary.json"
    shell:
        "./src/prism_qc/target/release/prism_qc_aggregator {input.manifest} results/qc {output.summary}"
