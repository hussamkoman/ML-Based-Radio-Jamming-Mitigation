import math
import numpy as np
import h5py
import os
import tensorflow as tf
from keras import layers, models
from keras import mixed_precision
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

from radioSimulator import RadioSimulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mixed precision setup for faster training on GPU
opt = Adam(learning_rate=0.001)
opt = mixed_precision.LossScaleOptimizer(opt)
mixed_precision.set_global_policy('mixed_float16')

# 16-QAM symbol constellation points for mapping and plotting
qam16 = np.array([complex(i, q) for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]])


def iq_to_class(y):
    """
    Convert I/Q pairs to nearest 16-QAM class index.

    Args:
        y: array of shape (N, 2) containing real and imag parts.

    Returns:
        Array of length N with integer class labels in [0, 15].
    """
    iq_complex = y[:, 0] + 1j * y[:, 1]
    # Compute distance to each constellation point
    distances = np.abs(iq_complex[:, None] - qam16[None, :])
    return np.argmin(distances, axis=1)


class QAMReceiverTrainer:
    """
    Load simulated QAM data from HDF5, build and train a neural receiver model.
    """

    def __init__(self, file_path, samples_per_symbol, M, N_symbols=16):
        """
        Initialize trainer by reading metadata and computing RRC taps.

        Args:
            file_path: Path to HDF5 file containing simulation groups.
            samples_per_symbol: Number of samples corresponding to one QAM symbol.
            M: Constellation size (e.g. 16).
            N_symbols: Number of symbols to display in plots (default 16).
        """
        self.file_path = file_path
        self.samples_per_symbol = samples_per_symbol
        self.M = M
        self.N_symbols = N_symbols
        self.model = None
        self.max_amp_x = None
        self.max_amp_y = None

        # Read file-level attributes for data rates and counts
        with h5py.File(self.file_path, 'r') as f:
            self._carrier_frequency = f.attrs.get("carrier_freq")
            self._data_rate = f.attrs.get("data_rate")
            self._duration = f.attrs.get("duration")
            # Count how many simulation groups exist
            self._n_sims = sum(1 for k in f.keys() if k.startswith("sim_"))
            self._sampling_rate = f.attrs["sampling_rate"]

            if self.samples_per_symbol is None:
                # Derive samples per symbol if not provided
                self.samples_per_symbol = int(
                    self._sampling_rate / (self._data_rate / np.log2(self.M))
                )

        # Compute number of symbols per simulation
        bits_per_symbol = int(np.log2(self.M))
        self.symbols_per_sim = int(self._data_rate * self._duration) // bits_per_symbol

        # Log key parameters
        print(
            f"\n\n"
            f"# carrier freq: {self._carrier_frequency} Hz\n"
            f"# sampling rate: {self._sampling_rate} Hz\n"
            f"# SPS: {self.samples_per_symbol}\n"
            f"# bits/symbol: {bits_per_symbol}\n"
            f"# data rate: {self._data_rate} bps\n"
            f"# duration: {self._duration} s\n"
            f"# n_sims: {self._n_sims}\n"
            f"# symbols/sim: {self.symbols_per_sim}\n"
            f"# N_symbols: {self.N_symbols}\n\n"
        )

        # Precompute RRC filter taps matching the simulator
        self._rrc_taps = None
        self._rrc_taps = RadioSimulation.design_rrc(
            self,
            bits_per_symbol=bits_per_symbol,
            sps=self.samples_per_symbol,
            alpha=0.35,
            span=8
        ).astype(np.float32)

        if self.samples_per_symbol is None:
            logger.error("samples_per_symbol is required but missing.")
            raise ValueError("samples_per_symbol must be provided.")
        else:
            logger.info(f"Samples per symbol: {self.samples_per_symbol}")

    def load_data(self, starting_sim=0, sim_num=None):
        """
        (Not used) Load and normalize a batch of simulations into memory.

        Args:
            starting_sim: Index of first simulation to load.
            sim_num: Number of simulations to load (None = all).

        Returns:
            Tuple (X, Y) of input blocks and complex symbol targets.
        """
        with h5py.File(self.file_path, 'r') as f:
            sim_keys = sorted(f.keys())
            if sim_num is not None:
                sim_keys = sim_keys[starting_sim:starting_sim + sim_num]

            x_list, y_list = [], []
            for name in sim_keys:
                # Retrieve transmitted symbols and received waveform
                qam_symbols = f[name]["qam_symbols"][:] * np.sqrt(10)
                received = f[name]["received_signal"][:]
                # Trim to exact symbol count
                n = len(qam_symbols)
                received = received[: n * self.samples_per_symbol]
                # Reshape to (n_symbols, sps)
                blocks = received.reshape(n, self.samples_per_symbol)
                # Split into I/Q channels if complex
                if np.iscomplexobj(blocks):
                    iq = np.stack([blocks.real, blocks.imag], axis=-1)
                else:
                    iq = blocks[..., np.newaxis]
                x_list.append(iq.astype("float32"))
                y_list.append(np.stack([qam_symbols.real, qam_symbols.imag], axis=-1))

            X = np.concatenate(x_list)
            Y = np.concatenate(y_list)

            # Normalize X by 3-sigma RMS for robustness
            self.max_amp_x = np.sqrt(np.mean(X ** 2)) * 3
            X /= self.max_amp_x
            self.max_amp_y = np.max(np.abs(Y))

            if sim_keys:
                logger.info(f"Loaded sims {sim_keys[0]} to {sim_keys[-1]}")
            else:
                logger.warning("No simulations loaded; check indices.")

            logger.info(f"Normalization factor (3-sigma): {self.max_amp_x:.4f}")
            return X, Y

    def make_dataset(self):
        """
        Create a tf.data.Dataset generator that yields one block-per-symbol.

        Does not load all simulations simultaneously; reads HDF5 groups on the fly.
        """

        def gen():
            with h5py.File(self.file_path, "r") as f:
                keys = sorted(f.keys())

                for k in keys:
                    # load QAM symbols and the 1-D complex filtered baseband
                    qam = f[k]["qam_symbols"][:] * np.sqrt(10)  # (n,)
                    filtered = f[k]["filtered_signal"][:]  # (n*sps,) complex64

                    n = len(qam)
                    # 1) group every sps samples into one block → shape (n, sps)
                    filtered = filtered.reshape(n, self.samples_per_symbol)

                    # 2) split into real (I) and imag (Q) channels → (n, sps, 2)
                    iq = np.stack([filtered.real, filtered.imag], axis=-1).astype("float32")

                    # labels
                    lbl = iq_to_class(np.stack([qam.real, qam.imag], axis=-1))

                    # optional: normalize each sim to 3-sigma
                    rms = np.sqrt(np.mean(iq ** 2)) * 3
                    iq /= rms

                    for block, c in zip(iq, lbl):
                        yield block, c

        spec = (
            tf.TensorSpec(shape=(self.samples_per_symbol, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
        return ds

    def build_model(self, input_shape):
        """
        Build a Keras model with convolutional, residual, and attention layers.

        Args:
            input_shape: Tuple for model input (sps, 2).
        """
        # inputs: pre-filtered IQ blocks of shape=(sps, 2)
        inputs = layers.Input(shape=input_shape, name="Received_IQ")
        x = inputs
        # Initial conv block
        x = layers.Conv1D(32, 7, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        # Residual blocks
        for _ in range(3):
            resid = x
            x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            # Match channels if needed
            if x.shape[-1] != resid.shape[-1]:
                resid = layers.Conv1D(64, 1, padding="same")(resid)
            x = layers.Add()([x, resid])
            x = layers.Activation("relu")(x)
        # Multi-head attention
        att = layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization()(x)
        # Global feature pooling
        a = layers.GlobalAveragePooling1D()(x)
        m = layers.GlobalMaxPooling1D()(x)
        x = layers.Concatenate()([a, m])
        # Classification head
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.M, activation="softmax", dtype="float32")(x)

        model = models.Model(inputs, outputs, name="qam_receiver")
        model.compile(optimizer=opt,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"]
                      )

        model.summary()
        self.model = model

    def create_callbacks(self):
        """
        Return a list of Keras callbacks:
          - EarlyStopping on val_loss
          - ReduceLROnPlateau
          - ModelCheckpoint for best val_accuracy
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'ml_v2/00_best_qam_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks

    def train(self, X, Y, epochs=100, batch_size=4, validation_split=0.2):
        """
        Train model on in-memory data.

        Splits X, Y into train/validation, then fits with callbacks.
        """
        # Convert to class labels
        Y_classes = iq_to_class(Y)

        # Train/validation split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y_classes, test_size=validation_split, random_state=42, stratify=Y_classes
        )

        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train model
        callbacks = self.create_callbacks()
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def train_stream(self, batch_size=16, epochs=100):
        """
        Train model using streaming HDF5 dataset.

        Reads from make_dataset(), shuffles and batches on the fly.
        """
        base_ds = self.make_dataset()

        bits_per_symbol = int(np.log2(self.M))
        symbols_per_sim = int(self._data_rate * self._duration) // bits_per_symbol
        total_symbols = self._n_sims * symbols_per_sim
        n_batches = math.ceil(total_symbols / batch_size)

        val_batches = max(1, int(0.1 * n_batches))  # 10% of the dataset for validation

        steps_per_epoch = n_batches - val_batches

        train_ds = (
            base_ds
            .shuffle(10_000)
            .repeat()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            base_ds
            .batch(batch_size)
            .take(val_batches)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_batches,
            callbacks=self.create_callbacks(),
            verbose=1
        )
        return history

    def plot_training_history(self, history, filename='plots_v2/training_history'):
        """
        Plot training and validation loss/accuracy side by side, save to file.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        os.makedirs('plots_v2', exist_ok=True)
        base_filename = filename
        ext = '.png'
        i = 0
        filename = f"{base_filename}{ext}"
        while os.path.exists(filename):
            i += 1
            filename = f"{base_filename}_{i}{ext}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_detailed(self, ds):
        """
        Evaluate model on a Dataset, compute ACC, SER, BER, and plot subset constellation.

        Returns:
            Tuple of (accuracy, symbol_error_rate, bit_error_rate).
        """
        # Random sample for evaluation
        all_true, all_pred = [], []

        # Loop over all batches in ds
        for x_batch, y_batch in ds.as_numpy_iterator():
            _, pred_idxs, _ = self.predict_symbols(x_batch, batch_size=len(x_batch))
            all_true.append(y_batch)
            all_pred.append(pred_idxs)

        # Concatenate
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        # Compute metrics
        accuracy = np.mean(all_pred == all_true)
        ser = np.mean(all_pred != all_true)
        ber = np.sum([bin(a ^ b).count('1') for a, b in zip(all_pred, all_true)]) / (len(all_true) * 4)

        logger.info(f"Eval → Acc={accuracy:.4f}, SER={ser:.4f}, BER={ber:.4f}")

        # Final constellation plot (first min(200, len) points)
        N = min(200, len(all_true))
        true_syms = qam16[all_true[:N]]
        pred_syms = qam16[all_pred[:N]]
        y_true_coords = np.stack([true_syms.real, true_syms.imag], axis=-1)
        y_pred_coords = np.stack([pred_syms.real, pred_syms.imag], axis=-1)

        self.plot_constellation(
            y_true_coords,
            y_pred_coords,
            title=f"Eval Stream: Acc={accuracy:.3f}, SER={ser:.3f}, BER={ber:.4f}"
        )

        return accuracy, ser, ber

    def plot_first_n(self, ds, n=10, filename='plots_v2/first10.png'):
        """
        Plot the first n true vs. predicted symbols from the dataset.

        Args:
            ds: tf.data.Dataset yielding (block, class_index).
            n: Number of examples to plot.
        """
        # Pull batches until we have >= n examples
        collected_x, collected_y = [], []
        for x_batch, y_batch in ds.as_numpy_iterator():
            collected_x.append(x_batch)
            collected_y.append(y_batch)
            if sum(x.shape[0] for x in collected_x) >= n:
                break

        X_all = np.concatenate(collected_x, axis=0)
        Y_all = np.concatenate(collected_y, axis=0)

        x_sub = X_all[:n]
        y_true_cls = Y_all[:n]  # these are class indices

        # Predict
        pred_syms, pred_indices, _ = self.predict_symbols(x_sub, batch_size=n)
        y_pred_coords = np.stack([pred_syms.real, pred_syms.imag], axis=-1)

        # Ground truth coords
        true_syms = qam16[y_true_cls]
        y_true_coords = np.stack([true_syms.real, true_syms.imag], axis=-1)

        # Plot
        self.plot_constellation(
            y_true_coords,
            y_pred_coords,
            title=f"First {n} Symbols: True vs Predicted",
            filename=filename
        )

    def plot_constellation(self, Y_true, Y_pred, title="Constellation", filename=None):
        """
        Plot true vs. predicted symbol coordinates over ideal 16-QAM grid.

        Args:
            Y_true: Array of shape (N, 2) for true I/Q.
            Y_pred: Array of shape (N, 2) for predicted I/Q.
            title: Plot title.
            filename: If given, save plot to this path.
        """
        # Plot ideal constellation points
        qam16_real = np.real(qam16)
        qam16_imag = np.imag(qam16)
        plt.scatter(qam16_real, qam16_imag, s=200, c='gray', marker='x',
                    linewidths=1, label='Ideal QAM16', zorder=1)

        # Plot true symbols
        plt.scatter(Y_true[:, 0], Y_true[:, 1], s=30, color='red',
                    label="True Symbols", alpha=0.6, zorder=2)

        # Plot predicted symbols
        plt.scatter(Y_pred[:, 0], Y_pred[:, 1], s=20, color='blue',
                    label="Predicted Symbols", alpha=0.6, zorder=3)

        # Styling
        plt.axhline(0, color='black', linewidth=1, alpha=0.3)
        plt.axvline(0, color='black', linewidth=1, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.title(title, fontsize=14)
        plt.xlabel("In-phase (I)", fontsize=12)
        plt.ylabel("Quadrature (Q)", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def predict_symbols(self, X_test, batch_size=None):
        """
        Run inference on a batch of IQ blocks and return complex symbols, indices, and probabilities.

        Returns:
            (symbols, indices, probabilities)
        """
        probs = self.model.predict(X_test, batch_size=batch_size)
        idxs = np.argmax(probs, axis=1)
        syms = qam16[idxs]
        return syms, idxs, probs

    def save_model(self, path="ml_v2/qam_receiver.keras"):
        """
        Save the Keras model to disk, creating directories as needed.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path="ml_v2/qam_receiver.keras"):
        """
        Load a Keras model from disk.
        """
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")

    def check_estimated_memory(self, batch_size=4, max_gb=8.0):
        """
        Estimate GPU memory usage for model parameters and one batch of inputs.

        If estimated usage exceeds max_gb, exit the program.
        """
        if self.model is None:
            return
        param_bytes = sum(np.prod(v.shape) for v in self.model.trainable_variables) * 2
        input_bytes = batch_size * np.prod(self.model.input_shape[1:]) * 2
        total_gb = float(param_bytes + input_bytes) / 1e9

        logger.info(f"Estimated memory usage: {total_gb:.4f} GB")
        print(f"[MEM] Estimated: {total_gb:.4f} GB")
        if total_gb > max_gb:
            print(f"[MEM] Exceeds {max_gb} GB — exiting.")
            exit(1)

    def sample_xy(self, n, batch_size):
        """
        Return exactly n examples from the streaming dataset, batched for inference.

        Args:
            n: Number of examples to take.
            batch_size: Batch size for the returned dataset.
        """
        return (
            self.make_dataset()
            .shuffle(1000)
            .take(n)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


# ─── CONFIG ───────────────────────────────────────────────
USE_EXISTING_MODEL = False
PLOT_MODEL = True
MODEL_FILE = 'ml_v2/004_trained_on_dataset_04.keras'

DATASET_FILE = 'dataset_v2/04_qam_simulations.h5'
# DATA_RATE = 64
# SAMPLING_RATE = 1_000_000
SPS = 2048
M = 16

BATCH_SIZE = 32
EPOCHS = 30


def main():
    # GPU setup
    logical_gpus = tf.config.list_logical_devices('GPU')
    logger.info(f"Logical GPUs: {logical_gpus}")
    device = logical_gpus[0].name if logical_gpus else '/CPU:0'
    logger.info(f"Using device: {device}")
    logger.info(f"Training on Dataset file: {DATASET_FILE}")

    trainer = QAMReceiverTrainer(
        file_path=DATASET_FILE,
        samples_per_symbol=SPS,
        M=M
    )

    with tf.device(device):
        for i in range(1):
            # Load and prepare data
            tf.keras.backend.clear_session()

            # X, Y = trainer.load_data(starting_sim=0 * 400, sim_num=400)
            # X, Y = trainer.load_data()

            if USE_EXISTING_MODEL and os.path.exists(MODEL_FILE):
                logger.info(f"Loading model from {MODEL_FILE} …")
                trainer.load_model(MODEL_FILE)
            else:
                logger.info("Building new model …")
                input_shape = (trainer.samples_per_symbol, 2)
                trainer.build_model(input_shape)

                logger.info("Training model …")
                history = trainer.train_stream(batch_size=BATCH_SIZE, epochs=EPOCHS)

                trainer.plot_training_history(history, f"plots_v2/{os.path.basename(MODEL_FILE).replace('.keras', '')}")

                logger.info(f"Saving model to {MODEL_FILE} …")
                trainer.save_model(MODEL_FILE)

                # Optional: plot model architecture
                if PLOT_MODEL:
                    try:
                        arch_plot_name = f"plots_v2/{os.path.basename(MODEL_FILE).replace('.keras', '_architecture.pdf')}"
                        plot_model(trainer.model, to_file=arch_plot_name, show_shapes=True, dpi=300)
                        logger.info(f"Model architecture saved to {arch_plot_name}")
                    except Exception as e:
                        logger.warning(f"Could not generate model plot: {e}")

            test_ds = trainer.sample_xy(n=400, batch_size=16)
            # Detailed evaluation
            trainer.evaluate_detailed(test_ds)
            plotName = f"plots_v2/first10_constellation_{i:02d}.png"
            trainer.plot_first_n(test_ds, n=10, filename=plotName)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
