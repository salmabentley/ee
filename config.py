import numpy as np
import tensorflow as tf
from qiskit_machine_learning.utils import algorithm_globals

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
algorithm_globals.random_seed = 42
