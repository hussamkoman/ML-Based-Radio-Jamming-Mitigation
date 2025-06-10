import tensorflow as tf
import sionna
import pyjama
import numpy as np

print("========== PYJAMA ENVIRONMENT CHECK ==========")

# Check TensorFlow
try:
    print(f"✅ TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✅ GPU devices: {gpus if gpus else '❌ No GPU found'}")
except Exception as e:
    print("❌ TensorFlow check failed:", e)

# Check Sionna
try:
    print(f"✅ Sionna version: {sionna.__version__}")
    from sionna.channel import FlatFadingChannel
    fc = FlatFadingChannel(1, 1, 1)
    print("✅ Sionna functional: FlatFadingChannel created")
except Exception as e:
    print("❌ Sionna test failed:", e)

# Check Pyjama
try:
    print(f"✅ Pyjama version: {pyjama.__version__}")
    from pyjama import simulation_model
    from sionna.ofdm import ResourceGrid
    from sionna.channel import FlatFadingChannel
    import tensorflow as tf

    # Params
    num_tx = 1
    num_tx_ant = 1
    num_rx = 1
    num_rx_ant = 1
    carrier_frequency = 3.5e9

    rg = ResourceGrid(fft_size=72, num_ofdm_symbols=14, num_tx=num_tx, subcarrier_spacing=15e3)
    channel = FlatFadingChannel(num_rx_ant*num_rx, num_tx_ant*num_tx, rg.num_ofdm_symbols)

    model = simulation_model.OFDMJammer(channel_model=channel, rg=rg, num_tx=num_tx, num_tx_ant=num_tx_ant)

    # Provide dummy input
    real = tf.random.normal((1, rg.num_ofdm_symbols, rg.fft_size, num_tx, num_tx_ant), dtype=tf.float32)
    imag = tf.random.normal((1, rg.num_ofdm_symbols, rg.fft_size, num_tx, num_tx_ant), dtype=tf.float32)
    x = tf.complex(real, imag)
    h = channel(x.shape[0])

    model((x, h), training=False)

    print("✅ simulate_model ran successfully")


except Exception as e:
    print("❌ Pyjama test failed:", e)

print("========== ✅ ENVIRONMENT OK ==========")
