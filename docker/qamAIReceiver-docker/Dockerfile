FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install basic tools
RUN apt-get update && apt-get install -y \
    curl ca-certificates git wget bzip2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Copy and create Conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Auto-activate the environment on login
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate qam-env" >> /root/.bashrc && \
    echo "force_color_prompt=yes" >> /root/.bashrc && \
    echo "PS1='(\$CONDA_DEFAULT_ENV) \[\033[01;32m\]\u@\h:\[\033[01;34m\]\w\$\[\033[00m\] '" >> /root/.bashrc

# Use the Conda environment by default
SHELL ["conda", "run", "-n", "qam-env", "/bin/bash", "-c"]

WORKDIR /app

#RUN conda install conda-forge::tensorflow-gpu
CMD ["bash"]