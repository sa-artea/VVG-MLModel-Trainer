# What version of Python do you have?
# upgrade pip packages with dependencies care
# pip freeze | %{$_.split('==')[0]} | %{pip install --upgrade --upgrade-strategy only-if-needed $_}
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print(tf.config.list_physical_devices('GPU'))
print("GPU is", "available" if gpu else "NOT AVAILABLE")
