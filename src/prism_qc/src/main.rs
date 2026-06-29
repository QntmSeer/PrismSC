use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModalityStats {
    n_cells_raw: usize,
    n_cells_filtered: usize,
    median_genes_per_cell: Option<f64>,
    median_peaks_per_cell: Option<f64>,
    median_counts_per_cell: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SampleQC {
    rna: Option<ModalityStats>,
    atac: Option<ModalityStats>,
}

#[derive(Serialize, Debug)]
struct CohortSummary {
    total_samples: usize,
    total_rna_cells_raw: usize,
    total_rna_cells_filtered: usize,
    total_atac_cells_raw: usize,
    total_atac_cells_filtered: usize,
    mean_rna_cells_per_sample: f64,
    cell_size_entropy: f64,
    sample_cell_counts: HashMap<String, usize>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: prism_qc_aggregator <manifest_tsv> <qc_dir> <output_json>");
        std::process::exit(1);
    }

    let manifest_path = &args[1];
    let qc_dir = &args[2];
    let output_path = &args[3];

    // 1. Read manifest TSV to extract sample IDs
    let manifest_file = match File::open(manifest_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening manifest: {}", e);
            std::process::exit(1);
        }
    };

    let reader = BufReader::new(manifest_file);
    let mut samples = Vec::new();
    
    // Skip header line and parse sample_id from first column
    for (idx, line) in reader.lines().enumerate() {
        if idx == 0 {
            continue; // Skip header
        }
        if let Ok(line_str) = line {
            let parts: Vec<&str> = line_str.split('\t').collect();
            if !parts.is_empty() && !parts[0].trim().is_empty() {
                samples.push(parts[0].trim().to_string());
            }
        }
    }

    let mut total_rna_raw = 0;
    let mut total_rna_filtered = 0;
    let mut total_atac_raw = 0;
    let mut total_atac_filtered = 0;
    let mut sample_counts = HashMap::new();

    // 2. Parse QC JSON files for each sample
    for sample in &samples {
        let qc_file_path = Path::new(qc_dir).join(format!("{}_qc_metrics.json", sample));
        if !qc_file_path.exists() {
            eprintln!("[WARN] QC metrics file for {} not found at {:?}", sample, qc_file_path);
            continue;
        }

        let file = match File::open(&qc_file_path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let reader = BufReader::new(file);
        let qc_data: Result<SampleQC, _> = serde_json::from_reader(reader);
        
        if let Ok(data) = qc_data {
            let mut sample_cells = 0;
            if let Some(rna) = data.rna {
                total_rna_raw += rna.n_cells_raw;
                total_rna_filtered += rna.n_cells_filtered;
                sample_cells += rna.n_cells_filtered;
            }
            if let Some(atac) = data.atac {
                total_atac_raw += atac.n_cells_raw;
                total_atac_filtered += atac.n_cells_filtered;
                if sample_cells == 0 {
                    sample_cells = atac.n_cells_filtered;
                }
            }
            sample_counts.insert(sample.clone(), sample_cells);
        }
    }

    // 3. Compute cohort balance statistics (Shannon Entropy)
    let total_filtered_cells: usize = sample_counts.values().sum();
    let mut entropy = 0.0;
    if total_filtered_cells > 0 {
        for &count in sample_counts.values() {
            let p = count as f64 / total_filtered_cells as f64;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
    }

    let mean_rna = if !samples.is_empty() {
        total_rna_filtered as f64 / samples.len() as f64
    } else {
        0.0
    };

    let summary = CohortSummary {
        total_samples: samples.len(),
        total_rna_cells_raw: total_rna_raw,
        total_rna_cells_filtered: total_rna_filtered,
        total_atac_cells_raw: total_atac_raw,
        total_atac_cells_filtered: total_atac_filtered,
        mean_rna_cells_per_sample: mean_rna,
        cell_size_entropy: entropy,
        sample_cell_counts: sample_counts,
    };

    // 4. Output summary to JSON
    let summary_json = serde_json::to_string_pretty(&summary).unwrap();
    let mut out_file = File::create(output_path).unwrap();
    out_file.write_all(summary_json.as_bytes()).unwrap();

    println!("[SUCCESS] Rust QC Aggregator consolidated metrics for {} samples.", samples.len());
}
