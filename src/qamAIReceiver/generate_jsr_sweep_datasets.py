# generate_jsr_sweep_datasets.py
import numpy as np
import h5py
import os
from radioSimulator import RadioSimulation, HDF5Writer

# Main parameters (same as your training) ===
M = 16  # QAM order
CARRIER_AMP = 1.0
FC = 10000
DATA_RATE = 64
SPS = 16_384
FS = DATA_RATE // np.log2(M) * SPS  # 4 Msps
DURATION = 1.0

# Barrage jammer parameters
JAM_BAND_WIDTH = 2000
TONE_SPACING = 100
JAM_JITTER = 30

JAM_START = FC - JAM_BAND_WIDTH // 2
JAM_END = FC + JAM_BAND_WIDTH // 2

SNR_LIST_dB = (20, 25, 30, 35, 40)
JSR_LIST_dB = [-10, -5, 0, 5, 10, 15, 20]

NUM_RUNS = 100  # simulations per JSR

OUTDIR = "dataset/jsr_sweep"

os.makedirs(OUTDIR, exist_ok=True)
GROUP_NAME = "qam_simulations"

rng = np.random.default_rng()

for jsr_dB in JSR_LIST_dB:
    output_file = f"{OUTDIR}/07_qam_simulations_jsr_{jsr_dB}.h5"
    # fresh file for each JSR
    with h5py.File(output_file, "w"):
        pass
    writer = HDF5Writer(output_file, GROUP_NAME)

    print(f"Generating {NUM_RUNS} simulations at JSR={jsr_dB} dB")

    for idx in range(NUM_RUNS):
        snr_dB = rng.choice(SNR_LIST_dB)
        noise_pow = CARRIER_AMP ** 2 / (2 * 10 ** (snr_dB / 10))
        jammer_pow = CARRIER_AMP ** 2 * 10 ** (jsr_dB / 10)

        jam_base = np.arange(JAM_START, JAM_END + 1, TONE_SPACING)
        jitter = rng.uniform(-JAM_JITTER, JAM_JITTER, size=jam_base.shape)
        jammer_freqs = jam_base + jitter
        closest_idx = np.argmin(np.abs(jammer_freqs - FC))
        jammer_freqs[closest_idx] = FC

        sim = RadioSimulation(
            carrier_amplitude=CARRIER_AMP, carrier_frequency=FC, duration=DURATION,
            data_rate=DATA_RATE, sampling_rate=FS,
            jammer_power=jammer_pow, jammer_frequencies=jammer_freqs, noise_power=noise_pow
        )
        sim.run_simulation()
        sim.add_barrage_jammer()
        writer.save_simulation(sim, idx)
        print(f"[{idx + 1:4d}/{NUM_RUNS}]  SNR={snr_dB:2d} dB  JSR={jsr_dB:2d} dB")

    print(f"Finished {output_file}")

print("All JSR sweep datasets generated in", OUTDIR)
