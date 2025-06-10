import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import CheckButtons
from scipy.signal import fftconvolve
import h5py


class RadioSimulation:
    """
    Simulate a QAM-based radio link with optional jamming and additive white Gaussian noise.
    """

    def __init__(
            self,
            tx_power: float,
            carrier_frequency: float,
            duration: float,
            data_rate: float,
            sampling_rate: float,
            noise_variance: float,
            load_resistance: float = 50.0,
            random_seed: int | None = None,
    ) -> None:
        """
        Initialize simulation parameters.

        Args:
            tx_power: Average transmit power in watts.
            carrier_frequency: Carrier frequency in hertz.
            duration: Total simulation time in seconds.
            data_rate: Bit rate in bits per second.
            sampling_rate: Sampling frequency in hertz (must be an integer multiple of data_rate).
            noise_variance: Variance of AWGN per sample.
            load_resistance: Load resistance in ohms (default 50).
            random_seed: Optional random seed for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self._tx_power = tx_power
        # Convert average power to peak amplitude: P_avg = A^2 / 2 => A = sqrt(2 * P_avg)
        self._carrier_amplitude = np.sqrt(2 * tx_power)
        self._carrier_frequency = carrier_frequency

        self._duration = duration
        self.load_resistance = load_resistance
        self._data_rate = data_rate
        self._sampling_rate = sampling_rate

        # Check that sampling_rate is integer multiple of data_rate
        sps_exact = sampling_rate / data_rate
        if not float(sps_exact).is_integer():
            raise ValueError(
                f"sampling_rate ({sampling_rate}) must be an integer multiple of data_rate ({data_rate})."
            )
        self._samples_per_bit = int(sps_exact)

        self._noise_variance = noise_variance

        # Precompute time vector (must produce integer number of samples)
        self._num_samples = int(duration * sampling_rate)
        self._time = np.arange(self._num_samples) / sampling_rate

        self._samples_per_symbol = None

        # Placeholders for signals
        self._binary_data: np.ndarray | None = None
        self._ook_data: np.ndarray | None = None
        self._qam_symbols: np.ndarray | None = None
        self._rrc_taps: np.ndarray | None = None
        self._modulated_signal: np.ndarray | None = None
        self._jammer: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._received_signal: np.ndarray | None = None
        self._filtered_signal: np.ndarray | None = None
        self._downsampled_symbols: np.ndarray | None = None

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def jammer(self) -> np.ndarray:
        return self._jammer

    @property
    def noise(self) -> np.ndarray:
        return self._noise

    @property
    def received_signal(self) -> np.ndarray:
        return self._received_signal

    @property
    def filtered_signal(self) -> np.ndarray:
        return self._filtered_signal

    def generate_data(self) -> None:
        """
        Generate a random bitstream and its on-off keying (OOK) waveform.

        Produces:
          _binary_data: raw bit array length = data_rate * duration
          _ook_data: each bit repeated samples_per_bit times, length = total samples
        """
        num_bits = int(self._data_rate * self._duration)
        self._binary_data = np.random.randint(0, 2, size=num_bits)

        # Build OOK data by repeating each bit
        repeated = np.repeat(self._binary_data, self._samples_per_bit)
        if len(repeated) < self._num_samples:
            self._ook_data = np.pad(repeated, (0, self._num_samples - len(repeated)))
        else:
            self._ook_data = repeated[:self._num_samples]

    def modulate_qam(self, M: int = 16, use_rrc=True) -> None:
        """
        Map binary data into M-QAM symbols with Gray coding, apply pulse shaping,
        normalize to unit power, then upconvert to a real carrier.

        Args:
            M: Constellation size (must be power of 2).
            use_rrc: If True, apply root-raised-cosine pulse shaping; else use rectangular pulses.
        """
        bits_per_symbol = int(np.log2(M))
        trimmed_len = (len(self._binary_data) // bits_per_symbol) * bits_per_symbol
        bits = self._binary_data[:trimmed_len].astype(int).reshape((-1, bits_per_symbol))

        # Split bits into I and Q groups, MSB to LSB
        bits_i = bits[:, : bits_per_symbol // 2]
        bits_q = bits[:, bits_per_symbol // 2:]

        # Convert each half‐bit‐group to integer 0,1,2,3
        weights = 1 << np.arange(bits_per_symbol // 2)[::-1]
        symbols_i = (bits_i * weights).sum(axis=1)
        symbols_q = (bits_q * weights).sum(axis=1)

        # Map standard 16-QAM Gray mapping {00,01,10,11} → {−3, −1, +1, +3}
        constellation_i = 2 * symbols_i - (np.sqrt(M) - 1)
        constellation_q = 2 * symbols_q - (np.sqrt(M) - 1)

        # Build complex symbols
        qam_symbols = constellation_i.astype(np.complex64) + 1j * constellation_q.astype(np.complex64)
        qam_symbols /= np.sqrt(10)  # normalize average symbol energy

        # Upsample QAM symbols by samples_per_symbol
        self._samples_per_symbol = self._samples_per_bit * bits_per_symbol
        num_symbols = len(qam_symbols)
        up_len = num_symbols * self._samples_per_symbol

        if use_rrc:
            upsampled = np.zeros(up_len, dtype=np.complex64)
            upsampled[:: self._samples_per_symbol] = qam_symbols

            # Design RRC (if not already done)
            if self._rrc_taps is None:
                self._rrc_taps = self.design_rrc(bits_per_symbol, self._samples_per_symbol, alpha=0.35, span=8)

            # rrc pulse-shape
            shaped = fftconvolve(upsampled, self._rrc_taps, mode="same").astype(np.complex64)
            shaped = shaped[:up_len]
        else:
            # rectangular: hold the symbol level
            shaped = np.repeat(qam_symbols, self._samples_per_symbol)

        # Normalize baseband waveform to unit power
        shaped /= np.sqrt(np.mean(np.abs(shaped) ** 2))

        # Carrier upconversion (complex baseband → real passband)
        t = self._time.astype(np.float32)
        carrier_exp = np.exp(2j * np.pi * self._carrier_frequency * t, dtype=np.complex64)

        # Note: carrier_amplitude = sqrt(2 * carrier_power)
        self._modulated_signal = (self._carrier_amplitude * np.real(shaped * carrier_exp)).astype(np.float32)

        self._qam_symbols = qam_symbols
        self._received_signal = self._modulated_signal.copy()

    def design_rrc(self, bits_per_symbol: int, sps: int, alpha: float, span: int) -> np.ndarray:
        """
        Design a root-raised-cosine (RRC) filter.

        Args:
            bits_per_symbol: Number of bits carried per QAM symbol.
            sps: Samples per symbol.
            alpha: Roll-off factor (0 < alpha <= 1).
            span: Filter span in symbol durations on each side.

        Returns:
            rrc_taps: Normalized filter taps with unit energy.
        """
        if self._rrc_taps is not None:
            return self._rrc_taps

        Ts = float(bits_per_symbol) / self._data_rate
        N = span * sps
        t_idx = np.arange(-N / 2, N / 2 + 1) / sps * Ts
        h = np.zeros_like(t_idx, dtype=np.float64)
        for i, t in enumerate(t_idx):
            if t == 0.0:
                h[i] = 1.0 - alpha + (4 * alpha / np.pi)
            elif abs(abs(t) - Ts / (4 * alpha)) < 1e-12:
                h[i] = (alpha / np.sqrt(2)) * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                numerator = np.sin(np.pi * t * (1 - alpha) / Ts) + 4 * alpha * t / Ts * np.cos(
                    np.pi * t * (1 + alpha) / Ts
                )
                denominator = np.pi * t * (1 - (4 * alpha * t / Ts) ** 2) / Ts
                h[i] = numerator / denominator
        # Normalize to unit energy
        h = h / np.sqrt(np.sum(h ** 2))
        self._rrc_taps = h.astype(np.float32)
        return self._rrc_taps

    def add_barrage_jammer(
            self,
            jammer_power: float,
            frequencies=None,
            *,
            dense: bool = False,
            num_tones: int = 128,
            tone_spacing: float = 1.0,
            smooth: bool = True,
    ) -> None:
        """
        Add a multi-tone barrage jammer signal.

        Args:
            jammer_power: Average jammer power in watts.
            frequencies: List of tone center frequencies in hertz (ignored if dense=True).
            dense: If True, generate an evenly spaced comb of tones around carrier.
            num_tones: Number of tones when dense=True.
            tone_spacing: Spacing in hertz between adjacent tones for dense comb.
            smooth: If True, apply a moving-average over the Rayleigh envelope (length floor(sps/4)).
        """
        t = self._time.astype(np.float32)

        if dense:
            fc = frequencies[0] if frequencies else self._carrier_frequency
            bins = (np.arange(num_tones) - num_tones // 2) * tone_spacing
            frequencies = (fc + bins).tolist()
        elif frequencies is None:
            self._jammer = np.zeros_like(t)
            return

        K = len(frequencies)
        if jammer_power <= 0 or K == 0:
            self._jammer = np.zeros_like(t)
            return

        # Rayleigh envelope (fast fading if smooth=False)
        env = np.random.rayleigh(np.sqrt(2 / np.pi), size=len(t)).astype(np.float32)
        if smooth and self._samples_per_bit > 4:
            L = int(self._samples_per_bit / 4)
            env = np.convolve(env, np.ones(L) / L, mode="same")

        # random phase per tone
        phases = np.random.uniform(0, 2 * np.pi, K).astype(np.float32)

        # build jammer signal
        jammer = np.zeros_like(t, dtype=np.float32)
        for f, phi in zip(frequencies, phases):
            jammer += env * np.sin(2 * np.pi * f * t + phi, dtype=np.float32)

        # Normalize to desired power
        jammer /= np.sqrt(np.mean(jammer ** 2))
        jammer *= np.sqrt(jammer_power)
        self._jammer = jammer.astype(np.float32)

        # Add to received buffer
        if self._modulated_signal is not None:
            self._received_signal += self._jammer
        else:
            self._received_signal = self._jammer.copy()

    def add_awgn(self) -> None:
        """
        Add zero-mean real AWGN to the received signal.

        Uses noise_variance per sample.
        """
        scale = np.sqrt(self._noise_variance)
        self._noise = np.random.normal(0.0, scale, size=self._num_samples).astype(np.float32)

        base = (
            getattr(self, "_received_signal", 0.0)
            if hasattr(self, "_received_signal")
            else np.zeros(self._num_samples, dtype=np.float32)
        )
        self._received_signal = (base + self._noise).astype(np.float32)

    def matched_filter(self) -> None:
        """
        Mix the received passband signal to complex baseband and apply matched filter.

        If RRC taps exist, use them; else use a rectangular (boxcar) filter.
        The output is stored in _filtered_signal.
        """
        if self._received_signal is None:
            raise RuntimeError("No received signal—run modulate_qam, add_jammer, or add_awgn first.")

        x = self._received_signal.astype(np.float32)
        t = self._time.astype(np.float32)
        sps = self._samples_per_symbol

        # I/Q mixing
        I_bb = x * np.cos(2 * np.pi * self._carrier_frequency * t, dtype=np.float32)
        Q_bb = x * -np.sin(2 * np.pi * self._carrier_frequency * t, dtype=np.float32)

        if self._rrc_taps is not None:
            # Matched RRC filter
            I_f = fftconvolve(I_bb, self._rrc_taps, mode="same").astype(np.float32)
            Q_f = fftconvolve(Q_bb, self._rrc_taps, mode="same").astype(np.float32)
        else:
            # fallback simple LP filter
            lp = np.ones(sps, dtype=np.float32) / sps
            I_f = fftconvolve(I_bb, lp, mode="same").astype(np.float32)
            Q_f = fftconvolve(Q_bb, lp, mode="same").astype(np.float32)

        self._filtered_signal = I_f.astype(np.complex64) + 1j * Q_f.astype(np.complex64)
        # Scale factor compensates for carrier amplitude and filter energy
        self._filtered_signal *= (2.0 / (self._carrier_amplitude * np.sqrt(sps)))

    def downsample(self) -> None:
        """
        Sample the matched-filter output at symbol centers without extra delay compensation.

        The result is stored in _downsampled_symbols.
        """
        if not hasattr(self, '_filtered_signal') or self._filtered_signal is None:
            raise RuntimeError("Run matched_filter() before downsampling.")

        sps = self._samples_per_symbol
        num_sym = len(self._qam_symbols)

        # Calculate the correct sample centers including filter delay
        centers = np.arange(num_sym) * sps

        mf = self._filtered_signal
        if centers[-1] >= len(mf):
            # pad if needed
            pad = centers[-1] - (len(mf) - 1)
            mf = np.pad(mf, (0, pad))

        I_samples = mf.real[centers]
        Q_samples = mf.imag[centers]

        self._downsampled_symbols = (I_samples + 1j * Q_samples).astype(np.complex64)

    def demodulate_qam(self, M: int = 16) -> np.ndarray:
        """
        Perform hard-decision demodulation of QAM symbols.

        Thresholds at +/- (2/sqrt(10)) bisect the 16-QAM grid.
        Returns a flat array of recovered bits.
        """
        # automatically run matched_filter + downsample if not done
        if not hasattr(self, '_filtered_signal') or self._filtered_signal is None:
            self.matched_filter()

        if not hasattr(self, '_downsampled_symbols') or self._downsampled_symbols is None:
            self.downsample()

        sym = self._downsampled_symbols
        bits_per_symbol = int(np.log2(M))
        thr = 2 / np.sqrt(10)

        I_vals = sym.real
        Q_vals = sym.imag

        # map to nearest constellation point
        idx_I = np.zeros_like(I_vals, dtype=int)
        idx_Q = np.zeros_like(Q_vals, dtype=int)

        # For I axis:
        idx_I[I_vals < -thr] = -3
        idx_I[(I_vals >= -thr) & (I_vals < 0)] = -1
        idx_I[(I_vals >= 0) & (I_vals < thr)] = +1
        idx_I[I_vals >= thr] = +3

        # For Q axis:
        idx_Q[Q_vals < -thr] = -3
        idx_Q[(Q_vals >= -thr) & (Q_vals < 0)] = -1
        idx_Q[(Q_vals >= 0) & (Q_vals < thr)] = +1
        idx_Q[Q_vals >= thr] = +3

        # convert to bit pairs
        bin_I = ((idx_I + 3) // 2).astype(int)
        bin_Q = ((idx_Q + 3) // 2).astype(int)
        weights = 1 << np.arange(bits_per_symbol // 2)[::-1]
        bits_I = ((bin_I.reshape(-1, 1) & weights) > 0).astype(int)
        bits_Q = ((bin_Q.reshape(-1, 1) & weights) > 0).astype(int)
        return np.hstack((bits_I, bits_Q)).flatten()

    def save_to_hdf5(
            self,
            filename: str,
            group_name: str | None = None,
            *,
            compact: bool = False,
            overwrite: bool = False,
    ) -> None:
        """
        Save simulation data to an HDF5 file.

        Args:
            filename: Target HDF5 file.
            group_name: Name of the group; auto-named as sim_0000, sim_0001, ... if None.
            compact: If True, store only received_signal and qam_symbols; else full dump.
            overwrite: If True, replace an existing group with the same name.
        """
        with h5py.File(filename, "a") as h5:
            # auto-name group if not supplied
            if group_name is None:
                n = 0
                while f"sim_{n:04d}" in h5:
                    n += 1
                group_name = f"sim_{n:04d}"

            if group_name in h5:
                if overwrite:
                    del h5[group_name]
                else:
                    raise ValueError(f"group '{group_name}' already exists")

            g = h5.create_group(group_name)

            # Always include symbols and filtered or raw received signal
            if self._filtered_signal is not None:
                g.create_dataset("filtered_signal", data=self._filtered_signal)
                if not compact:
                    g.create_dataset("received_signal", data=self._received_signal)
            else:
                g.create_dataset("received_signal", data=self._received_signal)

            g.create_dataset("qam_symbols", data=self._qam_symbols)

            # optional full dump
            if not compact:
                g.create_dataset("time", data=self._time)
                carrier = self._carrier_amplitude * np.sin(2 * np.pi * self._carrier_frequency * self._time)
                g.create_dataset("carrier_signal", data=carrier)
                if self._jammer is not None:
                    g.create_dataset("jammer", data=self._jammer)
                if self._noise is not None:
                    g.create_dataset("noise", data=self._noise)

            # Store metadata
            g.attrs.update({
                "carrier_freq_Hz": self._carrier_frequency,
                "sampling_rate_Hz": self._sampling_rate,
                "data_rate_bps": self._data_rate,
                "duration_s": self._duration,
                "tx_power_W": self._tx_power,
                "noise_variance": self._noise_variance,
            })
            if hasattr(self, "_jammer"):
                g.attrs["jammer_power_W"] = float(self._jammer.var())


class SignalPlotter:
    """
    Plot time-domain, frequency-domain, and constellation diagrams from a RadioSimulation.
    """

    def __init__(self, radio_sim: RadioSimulation) -> None:
        """
        Store reference to a RadioSimulation instance for plotting.
        """
        self.radio_simulation = radio_sim

    def plot_time_domain(self) -> None:
        """
        Create an interactive time-domain plot with checkboxes to toggle visibility of:
          - OOK waveform (magenta)
          - Matched-filter output (real part, gold)
          - Received passband signal (blue)
          - Modulated passband signal (orange)
          - Carrier sinusoid (green)
          - Jammer waveform (red)
          - AWGN noise (gray)
          - Transmit symbol samples (cyan)
          - Receive symbol samples (black)

        Checkbox labels are colored to match the trace color. By default, only
        matched-filter and received signals are shown.
        """
        t = self.radio_simulation._time

        # Safely fetch each signal, defaulting to zeros if None
        if self.radio_simulation._received_signal is not None:
            data = self.radio_simulation._ook_data
        else:
            data = np.zeros_like(t)

        if self.radio_simulation.received_signal is not None:
            received = self.radio_simulation.received_signal
        else:
            received = np.zeros_like(t)

        if self.radio_simulation.filtered_signal is not None:
            raw = self.radio_simulation.filtered_signal
            filtered = raw.real
        else:
            filtered = np.zeros_like(t)

        if self.radio_simulation._modulated_signal is not None:
            modulated = self.radio_simulation._modulated_signal
        else:
            modulated = np.zeros_like(t)

        sym_tx = np.zeros_like(t)
        sym_rx = np.zeros_like(t)

        sps = self.radio_simulation._samples_per_symbol

        if hasattr(self.radio_simulation, "_qam_symbols"):
            # use real part for time-plot (imag part would be another trace)
            sym_tx_full = np.repeat(self.radio_simulation._qam_symbols.real, sps)
            sym_tx[:len(sym_tx_full)] = sym_tx_full[:len(sym_tx)]

        if hasattr(self.radio_simulation, "_downsampled_symbols"):
            sym_rx_full = np.repeat(self.radio_simulation._downsampled_symbols.real, sps)
            sym_rx[:len(sym_rx_full)] = sym_rx_full[:len(sym_rx)]

        # carrier is always defined once modulate_qam() has run
        carrier = (self.radio_simulation._carrier_amplitude
                   * np.sin(2 * np.pi * self.radio_simulation._carrier_frequency * t))

        if self.radio_simulation._jammer is not None:
            jammer = self.radio_simulation._jammer
        else:
            jammer = np.zeros_like(t)

        if self.radio_simulation._noise is not None:
            noise = self.radio_simulation._noise
        else:
            noise = np.zeros_like(t)

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(left=0.25)  # room for checkboxes

        # Plot lines with explicit colors

        l_data, = ax.plot(t, data, label="Data", color="magenta", linewidth=0.6, visible=False, zorder=9)
        l_filtered, = ax.plot(t, filtered, label="Filtered Signal", color="gold", linewidth=0.6, zorder=6)
        l_received, = ax.plot(t, received, label="Received Signal", color="blue", linewidth=0.6, zorder=2)
        l_modulated, = ax.plot(t, modulated, label="Modulated Signal", color="orange", alpha=0.7, visible=False,
                               zorder=4)
        l_carrier, = ax.plot(t, carrier, label="Carrier", color="green", alpha=0.5, visible=False, zorder=1)
        l_jammer, = ax.plot(t, jammer, label="Jammer", color="red", alpha=0.7, visible=False, zorder=3)
        l_noise, = ax.plot(t, noise, label="Noise", color="gray", alpha=0.3, visible=False, zorder=5)
        l_txsym, = ax.plot(t, sym_tx, label="Tx Symbols", color="cyan", linewidth=0.8, visible=False, zorder=7)
        l_rxsym, = ax.plot(t, sym_rx, label="Rx Symbols", color="black", linewidth=0.8, visible=False, zorder=8)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time-Domain Signals")
        ax.grid(True)

        # Checkboxes
        labels = ["Data", "Filtered Signal", "Received Signal", "Modulated Signal", "Carrier", "Jammer", "Noise",
                  "Tx Symbols", "Rx Symbols"]
        visibility = [False, True, True, False, False, False, False, False, False]

        checkbox_ax = plt.axes([0.02, 0.3, 0.20, 0.30])  # [left, bottom, width, height]
        check = CheckButtons(checkbox_ax, labels, visibility)

        # Match label text colors to line colors
        series_colors = ["magenta", "gold", "blue", "orange", "green", "red", "gray", "cyan", "black"]
        for i, color in enumerate(series_colors):
            check.labels[i].set_color(color)

        lines = {
            "Data": l_data,
            "Filtered Signal": l_filtered,
            "Received Signal": l_received,
            "Modulated Signal": l_modulated,
            "Carrier": l_carrier,
            "Jammer": l_jammer,
            "Noise": l_noise,
            "Tx Symbols": l_txsym,
            "Rx Symbols": l_rxsym,
        }

        def on_toggle(label: str) -> None:
            lines[label].set_visible(not lines[label].get_visible())
            plt.draw()

        check.on_clicked(on_toggle)

        plt.show()

    def plot_frequency_domain(self, signal: np.ndarray, title: str = "Magnitude Spectrum",
                              show_positive: bool = True) -> None:
        """
        Plot the magnitude spectrum (in dB) of a real signal.

        Args:
            signal: 1D array of time-domain samples.
            title: Plot title.
            show_positive: If True, show only f >= 0; else show full two-sided spectrum.
        """
        fs = self.radio_simulation.sampling_rate
        N = len(signal)
        # Compute frequency axis and normalized magnitude
        freq_axis = np.fft.fftfreq(N, d=1.0 / fs)
        spec = np.abs(np.fft.fft(signal)) / N

        fig, ax = plt.subplots(figsize=(8, 4))
        if show_positive:
            idx = freq_axis >= 0
            ax.plot(freq_axis[idx], 20 * np.log10(spec[idx] + 1e-12))
        else:
            ax.plot(np.fft.fftshift(freq_axis), 20 * np.log10(np.fft.fftshift(spec) + 1e-12))

        ax.set_title(title)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.grid(True)
        plt.tight_layout()
        # plt.show()

    def plot_constellation(self, n: int = 10) -> None:
        """
        Plot a QAM constellation for the first n symbols:
          - Transmitted symbols (blue)
          - Received symbols (green)
          - Ideal grid points (gray crosses)

        All series can be toggled via checkboxes whose label text matches the point color.
        """
        # Ensure symbols exist
        if not hasattr(self.radio_simulation, "_qam_symbols") or self.radio_simulation._qam_symbols is None:
            raise RuntimeError("Run modulate_qam() first to generate _qam_symbols.")

        if (not hasattr(self.radio_simulation, "_downsampled_symbols")
                or self.radio_simulation._downsampled_symbols is None):
            raise RuntimeError("Run matched_filter_and_downsample() first to generate _downsampled_symbols.")

        # Recover and rescale first n symbols
        tx_syms = self.radio_simulation._qam_symbols[:n] * np.sqrt(10)  # complex64 array of transmitted symbols

        rx_syms = self.radio_simulation._downsampled_symbols[:n] * np.sqrt(10)  # complex64 array of received symbols

        # Build figure and main axes
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(left=0.3)  # make room on the left for checkboxes

        # Plot each series (initially visible)
        scat_tx = ax.scatter(
            tx_syms.real,
            tx_syms.imag,
            s=20,
            c="blue",
            alpha=0.7,
            label="Transmitted",
        )
        scat_rx = ax.scatter(
            rx_syms.real,
            rx_syms.imag,
            s=20,
            c="green",
            alpha=0.7,
            label="Received",
        )

        M = 16
        levels = np.array([-3, -1, +1, +3])
        ideal_I, ideal_Q = np.meshgrid(levels, levels)
        ideal_I = ideal_I.flatten()
        ideal_Q = ideal_Q.flatten()
        scat_ideal = ax.scatter(
            ideal_I,
            ideal_Q,
            c="gray",
            marker="x",
            s=50,
            alpha=0.3,
            label="Ideal",
        )

        #  Formatting
        ax.set_xlabel("In-Phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(f"16-QAM Constellation of {n} symbols")
        ax.grid(True)
        ax.set_aspect("equal", "box")

        # Checkboxes, with label text colored to match the point colors
        labels = ["Transmitted", "Received", "Ideal"]
        visibility = [True, True, True]
        checkbox_ax = plt.axes([0.02, 0.3, 0.25, 0.2])  # [left, bottom, width, height]
        check = CheckButtons(checkbox_ax, labels, visibility)

        # Only color the label texts, not the boxes
        series_colors = ["blue", "green", "gray"]
        for i, color in enumerate(series_colors):
            check.labels[i].set_color(color)

        scatters = {
            "Transmitted": scat_tx,
            "Received": scat_rx,
            "Ideal": scat_ideal,
        }

        def on_toggle(label: str) -> None:
            scatters[label].set_visible(not scatters[label].get_visible())
            plt.draw()

        check.on_clicked(on_toggle)

        # Display
        # plt.show()


def main() -> None:
    # sim parameters
    M = 16
    tx_power = 1.0  # W
    carrier_frequency = 10000  # Hz
    duration = 1  # seconds
    SPS = 1024  # samples per symbol
    data_rate = 64  # bits/sec
    sampling_rate = data_rate // np.log2(16) * SPS  # Hz (must be multiple of data_rate)

    # jammer setting-----------------------------------------------------------------------------------------
    jammer_power = 1.0  # total jammer power

    JAM_START = carrier_frequency - 250.0  # jammer band lower edge
    JAM_END = carrier_frequency + 250.0  # jammer band upper edge
    TONE_SPACING = 2.50  # nominal delta f between tones
    JAM_JITTER = 1.0  # +-Hz random offset

    rng = np.random.default_rng()

    jam_base = np.arange(JAM_START, JAM_END + TONE_SPACING, TONE_SPACING)
    jitter = rng.uniform(-JAM_JITTER, JAM_JITTER, size=jam_base.shape)
    jammer_frequencies = jam_base + jitter  # Hz
    jammer_frequencies[np.argmin(np.abs(jammer_frequencies - carrier_frequency))] = carrier_frequency

    noise_variance = 1e-2  # per-sample variance
    seed = 45

    sim = RadioSimulation(
        tx_power,
        carrier_frequency,
        duration,
        data_rate,
        sampling_rate,
        noise_variance,
        random_seed=seed,
    )

    # 1. Generate data
    sim.generate_data()

    # 2. Modulate QAM
    sim.modulate_qam(M=M, use_rrc=True)

    # sim.design_rrc(bits_per_symbol=np.sqrt(M), sps=SPS, alpha=0.35, span=8)
    # 3. Add barrage jammer
    sim.add_barrage_jammer(jammer_power=jammer_power, frequencies=jammer_frequencies, smooth=True)

    # 4. Add AWGN
    sim.add_awgn()

    # 5. Receiver: matched filter and downsample

    # 6. Demodulate and compute BER (example)
    rec_bits = sim.demodulate_qam(M=16)

    tx_bits = sim._binary_data[: len(rec_bits)]
    num_errors = np.sum(tx_bits != rec_bits)
    ber = num_errors / len(tx_bits)
    print(f"Bit Error Rate: {ber:.4e}")

    # 7. Save all data
    sim.save_to_hdf5("simulation_output.h5")

    # 8. Plot results
    plotter = SignalPlotter(sim)
    plotter.plot_frequency_domain(signal=sim.received_signal, title="Received Signal Spectrum", show_positive=True)
    plotter.plot_frequency_domain(signal=sim.jammer, title="Jammer Spectrum", show_positive=True)
    plotter.plot_constellation(n=10)
    plotter.plot_time_domain()


if __name__ == "__main__":
    main()
