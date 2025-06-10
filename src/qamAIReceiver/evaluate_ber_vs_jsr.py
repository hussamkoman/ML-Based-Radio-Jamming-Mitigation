import numpy as np
from receiverTrainingModel import QAMReceiverTrainer, iq_to_class
import h5py
import matplotlib.pyplot as plt

# JSR sweep values in dB
jsr_values = [-10, -5, 0, 5, 10, 15, 20]
ber_results = []

# Paths
MODEL_PATH = "ml/007_trained_on_dataset_07.keras"
DATASET_PATH_TEMPLATE = "dataset/jsr_sweep/07_qam_simulations_jsr_{jsr}.h5"
GROUP_NAME = "qam_simulations"
M = 16

for jsr in jsr_values:
    # Prepare dataset for this JSR (you need to generate these if not yet)
    dataset_path = DATASET_PATH_TEMPLATE.format(jsr=jsr)
    trainer = QAMReceiverTrainer(
        file_path=dataset_path,
        group_name="qam_simulations",
        samples_per_symbol=16_384,
        M=16
    )
    X, Y = trainer.load_data()
    trainer.load_model(MODEL_PATH)
    # Get predictions
    Y_classes = iq_to_class(Y)
    pred_probs = trainer.model.predict(X, verbose=0)
    pred_indices = np.argmax(pred_probs, axis=1)
    # Calculate BER
    bit_errors = np.sum([bin(a ^ b).count('1') for a, b in zip(Y_classes, pred_indices)])
    total_bits = len(Y_classes) * 4  # 4 bits per 16-QAM symbol
    ber = bit_errors / total_bits
    ber_results.append(ber)

# Plotting
plt.figure(figsize=(6, 4))
plt.semilogy(jsr_values, ber_results, marker='o', label='Neural Receiver')
plt.xlabel('JSR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. JSR')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('plots/jsr_sweep/007_ber_vs_jsr.pdf')
# plt.show()
