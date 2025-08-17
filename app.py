# ================================
# EE: Classical vs Quantum Regressor (pt1)
# with custom PQC ansatz + visualisations
# Qiskit V2 primitives compatible
# ================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# -----------------------
# Scikit-learn
# -----------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------
# TensorFlow (Classical NN)
# -----------------------
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras

# -----------------------
# Qiskit (Quantum NN)
# -----------------------
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator  # V2 primitive

algorithm_globals.random_seed = 42

# =======================
# Load & Prepare Dataset
# =======================
df = pd.read_csv("dielectron.csv")
df.columns = df.columns.str.strip()  # remove trailing/leading spaces

target_col = "pt1"
features = [
    "E1", "px1", "py1", "pz1", "eta1", "phi1",
    "E2", "px2", "py2", "pz2", "pt2", "eta2", "phi2",
    "Q1", "Q2"
]

X = df[features].values
y = df[[target_col]].values  # shape (n, 1)

# Standardize inputs
X = StandardScaler().fit_transform(X)

# Scale target to [0,1] for training stability
y_scaler = MinMaxScaler((0, 1))
y = y_scaler.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# =======================
# Quantum Neural Network
# =======================
n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=42)
X_train_low = pca.fit_transform(X_train)
X_test_low = pca.transform(X_test)

# Custom PQC ansatz: Ry → CX entanglement → Ry
params = ParameterVector("θ", length=2 * n_qubits)
qc = QuantumCircuit(n_qubits)

for i in range(n_qubits):
    qc.ry(params[i], i)

qc.cx(0, 1)
qc.cx(0, 2)
qc.cx(1, 3)
qc.cx(2, 3)

for i in range(n_qubits):
    qc.ry(params[n_qubits + i], i)

ansatz = qc.remove_final_measurements(inplace=False)

# Feature map + compose
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
qnn_circuit = feature_map.compose(ansatz, inplace=False)

# EstimatorQNN with StatevectorEstimator
estimator = StatevectorEstimator()

qnn = EstimatorQNN(
    circuit=qnn_circuit,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
    gradient="param_shift"  # string works in ML >=0.11
)

# Optimizer: L-BFGS-B
from functools import partial
from scipy.optimize import minimize

scipy_optimizer = partial(minimize, method="L-BFGS-B", options={"maxiter": 20})

regressor = NeuralNetworkRegressor(
    neural_network=qnn,
    optimizer=scipy_optimizer
)

# Train
start_qnn = time.time()
regressor.fit(X_train_low, y_train.ravel())
train_time_qnn = time.time() - start_qnn

# Predict
y_pred_qnn = regressor.predict(X_test_low)
y_pred_qnn_rescaled = y_scaler.inverse_transform(y_pred_qnn.reshape(-1, 1))


# =======================
# Classical Neural Network
# =======================
input_layer = keras.Input(shape=(X_train.shape[1],))
x = keras.layers.Dense(64, activation="relu")(input_layer)
x = keras.layers.Dense(32, activation="relu")(x)
output = keras.layers.Dense(1, activation="linear")(x)
classical_model = keras.Model(inputs=input_layer, outputs=output)

classical_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        loss="mse", metrics=["mae"])

start_cls = time.time()
history = classical_model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=50, batch_size=32, verbose=1
)
train_time_classical = time.time() - start_cls

y_pred_classical = classical_model.predict(X_test, verbose=0)
y_test_rescaled = y_scaler.inverse_transform(y_test)
y_pred_classical_rescaled = y_scaler.inverse_transform(y_pred_classical)


# =======================
# Evaluation Metrics
# =======================
y_true = y_test_rescaled.flatten()
y_hat_cls = y_pred_classical_rescaled.flatten()
y_hat_qnn = y_pred_qnn_rescaled.flatten()

mse_cls = mean_squared_error(y_true, y_hat_cls)
mae_cls = mean_absolute_error(y_true, y_hat_cls)
r2_cls  = r2_score(y_true, y_hat_cls)

mse_qnn = mean_squared_error(y_true, y_hat_qnn)
mae_qnn = mean_absolute_error(y_true, y_hat_qnn)
r2_qnn  = r2_score(y_true, y_hat_qnn)

print("\n=== Classical NN Performance ===")
print(f"MSE: {mse_cls:.6f}")
print(f"MAE: {mae_cls:.6f}")
print(f"R² : {r2_cls:.6f}")
print(f"Train time: {train_time_classical:.3f} s")

print("\n=== Quantum NN Performance ===")
print(f"MSE: {mse_qnn:.6f}")
print(f"MAE: {mae_qnn:.6f}")
print(f"R² : {r2_qnn:.6f}")
print(f"Train time: {train_time_qnn:.3f} s")

# =======================
# Visualisations
# =======================

# 1) Circuit visualization (feature map + ansatz)
plt.figure()
qnn_circuit.draw("mpl")
plt.title("QNN Circuit: ZZFeatureMap ∘ Custom Ry–CX–Ry Ansatz")
plt.tight_layout()
plt.show()

# 2) Predictions vs True (scatter)
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_hat_cls, alpha=0.5, label="Classical NN", s=20)
plt.scatter(y_true, y_hat_qnn, alpha=0.5, label="Quantum NN", s=20)
mn, mx = np.min(y_true), np.max(y_true)
plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, label="Perfect Prediction")
plt.xlabel("True pt1")
plt.ylabel("Predicted pt1")
plt.title("Predictions vs True Values")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Error distributions
errors_classical = y_hat_cls - y_true
errors_qnn = y_hat_qnn - y_true

plt.figure(figsize=(8,6))
plt.hist(errors_classical, bins=50, alpha=0.6, label="Classical NN")
plt.hist(errors_qnn, bins=50, alpha=0.6, label="Quantum NN")
plt.xlabel("Prediction Error (pt1 units)")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# 4) Classical training curves
plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Classical NN Training Performance")
plt.legend()
plt.tight_layout()
plt.show()

# 5) Metric bar chart (MSE, MAE, R²)
labels = ["Classical", "Quantum"]
mse_vals = [mse_cls, mse_qnn]
mae_vals = [mae_cls, mae_qnn]
r2_vals  = [r2_cls, r2_qnn]

x = np.arange(len(labels))
w = 0.25

plt.figure(figsize=(9,6))
plt.bar(x - w, mse_vals, width=w, label="MSE")
plt.bar(x,     mae_vals, width=w, label="MAE")
plt.bar(x + w, r2_vals,  width=w, label="R²")
plt.xticks(x, labels)
plt.title("Model Metrics Comparison")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# 6) Training time bar chart
plt.figure(figsize=(7,5))
plt.bar(["Classical", "Quantum"], [train_time_classical, train_time_qnn])
plt.title("Training Time Comparison")
plt.ylabel("Seconds")
plt.tight_layout()
plt.show()
