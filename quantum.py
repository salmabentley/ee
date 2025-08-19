# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
# from qiskit.circuit.library import ZZFeatureMap, TwoLocal
# from qiskit.primitives import Estimator
# from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
# from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
# from qiskit_algorithms.optimizers import SPSA
# import joblib
# import os

# def build_and_train_qnn(
#     X_train, y_train,
#     n_qubits=4, reps=3, maxiter=100, seed=42,
#     save_path="qnn_model.pkl"
# ):
#     rng = np.random.default_rng(seed)

#     # 1. Initialize the scaler for the QNN's output range
#     y_scaler_qnn = MinMaxScaler(feature_range=(-1, 1))
    
#     # 2. FIT the scaler on the original training data BEFORE scaling
#     # This is the crucial step to ensure the scaler knows the true min/max.
#     y_scaler_qnn.fit(y_train.reshape(-1, 1))
    
#     # 3. TRANSFORM the training data for the QNN
#     y_scaled = y_scaler_qnn.transform(y_train.reshape(-1, 1)).ravel()

#     # Reduce dimensions using PCA
#     pca = PCA(n_components=n_qubits, random_state=seed)
#     X_low = pca.fit_transform(X_train)

#     # Scale inputs to angle range [0, 2pi]
#     x_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
#     X_embed = x_scaler.fit_transform(X_low)

#     # Quantum circuit
#     feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
#     ansatz = TwoLocal(n_qubits, ["ry", "rz"], entanglement="linear", reps=reps)
#     circuit = feature_map.compose(ansatz, inplace=False)

#     estimator = Estimator()
#     gradient = ParamShiftEstimatorGradient(estimator)

#     qnn = EstimatorQNN(
#         circuit=circuit,
#         input_params=feature_map.parameters,
#         weight_params=ansatz.parameters,
#         estimator=estimator,
#         gradient=gradient,
#     )

#     w0 = rng.normal(0, 0.1, size=qnn.num_weights)
#     optimizer = SPSA(maxiter=maxiter)

#     regressor = NeuralNetworkRegressor(
#         neural_network=qnn,
#         optimizer=optimizer,
#         initial_point=w0,
#         loss="squared_error",
#     )
#     regressor.fit(X_embed, y_scaled)

#     pipeline = {
#         "model": regressor,
#         "pca": pca,
#         "x_scaler": x_scaler,
#         "y_scaler_qnn": y_scaler_qnn,
#     }

#     if save_path:
#         joblib.dump(pipeline, save_path)
#         print(f"[INFO] QNN pipeline saved to {save_path}")

#     return pipeline

# def qnn_predict(pipeline, X):
#     X_low = pipeline["pca"].transform(X)
#     X_embed = pipeline["x_scaler"].transform(X_low)
#     y_scaled = pipeline["model"].predict(X_embed).reshape(-1, 1)
#     # The inverse transformation should now work correctly
#     return pipeline["y_scaler_qnn"].inverse_transform(y_scaled).ravel()

# def load_qnn_pipeline(path="qnn_model.pkl"):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"No saved QNN pipeline found at {path}")
#     pipeline = joblib.load(path)
#     print(f"[INFO] QNN pipeline loaded from {path}")

#     return pipeline

# # # Note: The `load_and_preprocess` function is assumed to return `y_scaler` for the classical model. 
# # # The `y_scaler` passed in your original code snippet is no longer necessary for the QNN training itself, 
# # # as the QNN will use its own dedicated scaler.

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.gradients.param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import L_BFGS_B # Use a Qiskit-native optimizer
import joblib
import os
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def build_and_train_qnn(
    X_train, y_train,
    n_qubits=4,
    reps=2,
    maxiter=100, # Increased maxiter
    seed=42,
    save_path="qnn_model.pkl", # New model file
):
    rng = np.random.default_rng(seed)

    subset_size = 300
    X_train_sub = X_train[:subset_size]
    y_train_sub = y_train[:subset_size]

    y_scaler_qnn = MinMaxScaler(feature_range=(-1, 1))
    y_scaler_qnn.fit(y_train.reshape(-1, 1))
    y_scaled = y_scaler_qnn.transform(y_train_sub.reshape(-1, 1)).ravel()

    pca = PCA(n_components=n_qubits, random_state=seed)
    pca.fit(X_train)
    X_low = pca.transform(X_train_sub)

    x_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    x_scaler.fit(pca.transform(X_train))
    X_embed = x_scaler.transform(X_low)

    params = ParameterVector("θ", length=2*n_qubits)

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):

        qc.ry(params[i], i)

        qc.cx(0,1); qc.cx(0,2); qc.cx(1,3); qc.cx(2,3)

    for i in range(n_qubits):

        qc.ry(params[n_qubits+i], i)

    ansatz = qc.remove_final_measurements(inplace=False)



    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)

    circuit = feature_map.compose(ansatz, inplace=False)

    estimator = Estimator()
    # gradient = ParamShiftEstimatorGradient(estimator)
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        # gradient=gradient
    )

    initial_point = rng.normal(0, 0.1, size=qnn.num_weights)
    
    # NEW: Use COBYLA
    optimizer = L_BFGS_B(maxiter=maxiter)

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        optimizer=optimizer,
        initial_point=initial_point,
        loss="squared_error",
    )
    regressor.fit(X_embed, y_scaled)

    pipeline = {
        "model": regressor,
        "pca": pca,
        "x_scaler": x_scaler,
        "y_scaler_qnn": y_scaler_qnn,
    }

    if save_path:
        joblib.dump(pipeline, save_path)
        print(f"[INFO] QNN pipeline saved to {save_path}")

    return pipeline

def qnn_predict(pipeline, X):
    # This function now correctly uses the PCA object from the pipeline
    X_low = pipeline["pca"].transform(X)
    X_embed = pipeline["x_scaler"].transform(X_low)
    y_scaled = pipeline["model"].predict(X_embed).reshape(-1, 1)
    print("QNN prediction (scaled) min/max:", y_scaled.min(), y_scaled.max())
    y_pred = pipeline["y_scaler_qnn"].inverse_transform(y_scaled).ravel()
    print("QNN prediction (rescaled) min/max:", y_pred.min(), y_pred.max())
    return pipeline["y_scaler_qnn"].inverse_transform(y_scaled).ravel()

def load_qnn_pipeline(path="qnn_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved QNN pipeline found at {path}")
    pipeline = joblib.load(path)
    print(f"[INFO] QNN pipeline loaded from {path}")
    return pipeline