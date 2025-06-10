FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# Install basic packages
RUN apt-get update && apt-get install -y \
    curl ca-certificates gnupg git wget bzip2 make \
    && apt-get clean


# Add CUDA APT pin priority
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin | \
    tee /etc/apt/preferences.d/cuda-repository-pin-600

# Install NVIDIA's CUDA keyring
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    rm cuda-keyring.deb

# Manually install specific libcublas version to satisfy dependencies
RUN apt-get update && apt-get install -y \
    libcublas-12-2=12.2.5.6-1 \
    && apt-mark hold libcublas-12-2

# Update APT again with NVIDIA repo now added
RUN apt-get update

# Install CUDA-related packages
RUN apt-get install -y \
    git \
    cuda-libraries-12-2 \
    libcudnn8 \
    libcudnn8-dev \
    libcurand-dev-12-2 \
    libcusolver-dev-12-2 \
    libcusparse-dev-12-2 \
    libcufft-dev-12-2 \
    llvm-15 \
    llvm-15-dev \
    && apt-get clean


# Set environment variables for GPU + Dr.Jit
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV DRJIT_LIBLLVM_PATH=/usr/lib/llvm-15/lib/libLLVM-15.so

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create and activate conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Set environment
ENV CONDA_DEFAULT_ENV=pyjama-env
ENV PATH=$CONDA_DIR/envs/pyjama-env/bin:$PATH

# Make Conda auto-activate and enable colored prompt
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate pyjama-env" >> /root/.bashrc && \
    echo "force_color_prompt=yes" >> /root/.bashrc && \
    echo "PS1='(\$CONDA_DEFAULT_ENV) \[\033[01;32m\]\u@\h:\[\033[01;34m\]\w\$\[\033[00m\] '" >> /root/.bashrc

# Set working directory
WORKDIR /app

# Install make if needed
RUN apt-get update && apt-get install -y make

# Clone Pyjama into isolated directory
RUN git clone https://github.com/hussamkoman/pyjama.git /opt/pyjama && \
    pip install -e /opt/pyjama &&\
    pip install /opt/pyjama[docs]


EXPOSE 8888
CMD [ "bash" ]
