import tensorflow as tf
import numpy as np
import scipy
import h5py
import sklearn

print("✅ TensorFlow version:", tf.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ SciPy version:", scipy.__version__)
print("✅ h5py version:", h5py.__version__)
print("✅ scikit-learn version:", sklearn.__version__)

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {gpus}")
else:
    print("❌ GPU NOT found")

