# ML-Based Radio Jamming Mitigation

A collection of Python scripts and tools for simulating, training, and evaluating QAM radio receivers under barrage
jamming, leveraging TensorFlow for robust anti-jamming techniques.

## Overview

This toolkit enables:

- **Simulation** of QAM transmitters and various barrage jamming attacks.
- **Dataset generation** (mixed jamming styles and JSR sweep HDF5 datasets).
- **Deep learning training** of a residual-attention QAM receiver using TensorFlow.
- **Performance evaluation** via BER vs. JSR curves and real-time inference benchmarks.
- **Containerized environment** powered by the `qamaireceiver` Docker container.

## Repository Structure

```
├── docker/                         # Dockerfile, docker-compose, and environment configs
├── src/
│   ├── check_env.py                # Validate CUDA, TensorFlow-GPU, Sionna & Mitsuba setup
│   ├── radioSimulator.py           # QAM transmitter + jamming models
│   ├── dataset_generator.py        # Generate mixed-style jamming HDF5 datasets
│   ├── generate_jsr_sweep_datasets.py # Create JSR sweep datasets
│   ├── receiverTrainingModel.py    # Define and train the anti-jamming receiver
│   ├── evaluate_ber_vs_jsr.py      # Compute and plot BER vs. JSR curves
│   └── inference_benchmark.py      # Measure inference latency & throughput
└── README.md                       # This file
```

## Prerequisites

- **Docker & Docker Compose**
- **NVIDIA GPU** with CUDA drivers (e.g., RTX 3070)
- **Python 3.9+** (if running scripts locally)
- **Conda** (optional for local environment)
- **Dependencies**: TensorFlow-GPU, h5py, NumPy, Matplotlib, Scikit-learn

## Quickstart with `qamaireceiver` Container

1. **Build and run**
   ```bash
   cd docker
   docker-compose up -d --build
   ```

2. **Enter container shell**
   ```bash
   docker-compose exec qamaireceiver bash
   ```

3. **Validate setup**
   ```bash
   python src/check_env.py
   ```

## Usage

Within the container or local environment, run:

- **Simulate QAM + Jamming**
  ```bash
  python src/radioSimulator.py
  ```

- **Generate Mixed Jamming Dataset**
  ```bash
  python src/dataset_generator.py
  ```

- **Generate JSR Sweep Dataset**
  ```bash
  python src/generate_jsr_sweep_datasets.py
  ```

- **Train Receiver**
  ```bash
  python src/receiverTrainingModel.py
  ```

- **Evaluate BER vs. JSR**
  ```bash
  python src/evaluate_ber_vs_jsr.py
  ```

- **Benchmark Inference**
  ```bash
  python src/inference_benchmark.py
  ```

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features.

## License

MIT License © FH Technikum Wien – Electrical Engineering
