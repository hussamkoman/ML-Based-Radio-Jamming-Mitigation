# dataset_generator.py
import os
import numpy as np
import h5py
from radioSimulator import RadioSimulation, SignalPlotter

# master parameters
OUT_FILE = "dataset_v2/04_qam_simulations.h5"
NUM_RUNS = 1
M = 16  # QAM order
FC = 100_000  # Hz
DATA_RATE = 8192  # bit/s
SPS = 2048  # samples per symbol
FS = DATA_RATE // np.log2(M) * SPS
DURATION = 1.0  # s
TX_POWER_W = 1

# jammer set-up
JAM_BW = 2_000  # Hz (total)
TONE_SPACING = 100  # Hz
JAM_JITTER = 30  # +-Hz random
JAM_START = FC - JAM_BW // 2
JAM_END = FC + JAM_BW // 2

SNR_dB = (20, 25, 30, 35, 40)
JSR_dB = (19, 20)

rng = np.random.default_rng()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# fresh file
with h5py.File(OUT_FILE, "w") as f:
    f.attrs.update({
        "M": M,
        "carrier_freq": FC,
        "data_rate": DATA_RATE,
        "sampling_rate": FS,
        "duration": DURATION,
        "tx_power_W": TX_POWER_W,
        "jam_bw_Hz": JAM_BW,
        "tone_spacing": TONE_SPACING,
    })

print("Generating", NUM_RUNS, "runs →", OUT_FILE)
print(f"  M               : {M}")
print(f"  TX-power [W]    : {TX_POWER_W}")
print(f"  Carrier freq [Hz]: {FC}")
print(f"  Data-rate [bit/s]: {DATA_RATE}")
print(f"  Sampling rate   : {FS}")
print(f"  Duration [s]    : {DURATION}")
print(f"  Jammer BW [Hz]  : {JAM_BW}")
print(f"  Tone spacing [Hz]: {TONE_SPACING}")
print()  # blank line before the per-run prints

sim = None

for idx in range(NUM_RUNS):
    snr_dB = rng.choice(SNR_dB)
    jsr_dB = rng.choice(JSR_dB)

    noise_var = TX_POWER_W / 10 ** (snr_dB / 10)  # AWGN variance
    jammer_pow = TX_POWER_W * 10 ** (jsr_dB / 10)  # jammer total power

    # jittered multi-tone list (always one tone on-carrier)
    jam_base = np.arange(JAM_START, JAM_END + 1, TONE_SPACING)
    jitter = rng.uniform(-JAM_JITTER, JAM_JITTER, jam_base.shape)
    jam_freqs = jam_base + jitter
    jam_freqs[np.argmin(np.abs(jam_freqs - FC))] = FC

    # build and run simulation
    sim = RadioSimulation(
        tx_power=TX_POWER_W,
        carrier_frequency=FC,
        duration=DURATION,
        data_rate=DATA_RATE,
        sampling_rate=FS,
        noise_variance=noise_var,
    )

    sim.generate_data()

    sim.modulate_qam(M, use_rrc=True)

    sim.add_awgn()

    # choose jammer style
    sim.add_barrage_jammer(jammer_power=jammer_pow,
                           frequencies=jam_freqs.tolist(),
                           smooth=False)
    jam_desc = "multi-tone jittered"

    sim.matched_filter()

    rx_bits = sim.demodulate_qam(M)
    tx_bits = sim._binary_data[: len(rx_bits)]
    ber = np.mean(rx_bits != tx_bits)

    # save
    gname = f"sim_{idx:04d}"
    sim.save_to_hdf5(OUT_FILE, group_name=gname, compact=True)

    with h5py.File(OUT_FILE, "a") as f:
        g = f[gname]
        g.attrs["BER"] = ber
        g.attrs["SNR_dB"] = snr_dB
        g.attrs["JSR_dB"] = jsr_dB
        g.attrs["jammer_power_W"] = jammer_pow
        g.attrs["jammer_type"] = jam_desc
        g.attrs["tone_count"] = len(jam_freqs)
    print(
        f"[{idx + 1:4d}/{NUM_RUNS}]  BER={ber:.4e}  SNR={snr_dB:2d}dB  JSR={jsr_dB:3d}dB  {jam_desc} noise var={noise_var:.4e}")

# plot last
plotter = SignalPlotter(sim)
plotter.plot_frequency_domain(signal=sim.received_signal, title="Received Signal Spectrum", show_positive=True)

plotter.plot_frequency_domain(signal=sim.jammer, title="Jammer Spectrum", show_positive=True)
plotter.plot_constellation(n=10)
plotter.plot_time_domain()
print("Done.")
