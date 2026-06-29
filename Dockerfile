FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file
COPY envs/clinical-sc-omics.yaml /tmp/environment.yaml

# Create environment
RUN conda env create -f /tmp/environment.yaml

# Activate environment by default
RUN echo "source activate clinical-sc-omics" > ~/.bashrc
ENV PATH /opt/conda/envs/clinical-sc-omics/bin:$PATH

# Copy project files
COPY . /app

# Default command
CMD ["snakemake", "--help"]
