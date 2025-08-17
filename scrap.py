# import time
# import numpy as np
# from qiskit import QuantumCircuit
# from qiskit.circuit import ParameterVector
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
# from qiskit_machine_learning.gradients.param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
# from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
# from sklearn.decomposition import PCA
# from functools import partial
# from scipy.optimize import minimize

# def build_and_train_qnn(
#     X_train, y_train,
#     n_qubits=4,
#     maxiter=20,
#     shots=1024,
#     token='DZEAT6chU3m01VL0QvIas1os8ZJR8e-yGKK7AjucQFAX',
#     channel="ibm_quantum_platform"
# ):
#     """
#     Train a QNN using IBM Quantum Runtime.
    
#     Parameters:
#         X_train, y_train : training data
#         n_qubits : number of qubits
#         maxiter : optimizer iterations
#         shots : number of measurement shots
#         token : IBM Quantum access token
#         channel : IBM Quantum channel
#     """
#     # Initialize IBM Quantum Runtime service
#     service = QiskitRuntimeService(channel=channel, token=token)

#     # Select backend (simulator or real device)
#     backend = service.backend("ibmq_qasm_simulator")
#     print(f"Using backend: {backend.name()}")

#     # PCA for dimensionality reduction
#     pca = PCA(n_components=n_qubits, random_state=42)
#     X_train_low = pca.fit_transform(X_train)

#     # PQC ansatz
#     params = ParameterVector("θ", length=2*n_qubits)
#     qc = QuantumCircuit(n_qubits)
#     for i in range(n_qubits):
#         qc.ry(params[i], i)
#     qc.cx(0,1); qc.cx(0,2); qc.cx(1,3); qc.cx(2,3)
#     for i in range(n_qubits):
#         qc.ry(params[n_qubits+i], i)
#     ansatz = qc.remove_final_measurements(inplace=False)

#     # Feature map
#     feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
#     qnn_circuit = feature_map.compose(ansatz, inplace=False)

#     # Estimator + gradient using IBM Quantum Runtime backend
#     estimator = Estimator(mode=backend)
#     gradient = ParamShiftEstimatorGradient(estimator)

#     # QNN
#     qnn = EstimatorQNN(
#         circuit=qnn_circuit,
#         input_params=feature_map.parameters,
#         weight_params=ansatz.parameters,
#         estimator=estimator,
#         gradient=gradient
#     )

#     # Callback to monitor training
#     iteration = {"count": 0}
#     def callback(xk):
#         iteration["count"] += 1
#         y_pred = qnn.forward(X_train_low, xk)
#         loss = np.mean((y_pred - y_train.ravel())**2)
#         print(f"Iteration {iteration['count']}, loss = {loss:.6f}")

#     optimizer = partial(
#         minimize,
#         method="L-BFGS-B",
#         options={"maxiter": maxiter},
#         callback=callback
#     )

#     # Train the regressor
#     start_time = time.time()
#     regressor = NeuralNetworkRegressor(neural_network=qnn, optimizer=optimizer)
#     regressor.fit(X_train_low, y_train.ravel())
#     train_time = time.time() - start_time

#     return regressor, pca, train_time


# import time
# import numpy as np
# from qiskit import QuantumCircuit
# from qiskit.circuit import ParameterVector
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
# from qiskit.primitives import Estimator
# from qiskit_machine_learning.gradients.param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
# from sklearn.decomposition import PCA
# from functools import partial
# from scipy.optimize import minimize

# def build_and_train_qnn(X_train, y_train, n_qubits=4, maxiter=3):
#     # PCA for dimensionality reduction
#     pca = PCA(n_components=n_qubits, random_state=42)
#     X_train_low = pca.fit_transform(X_train)

#     # PQC ansatz
#     params = ParameterVector("θ", length=2*n_qubits)
#     qc = QuantumCircuit(n_qubits)
#     for i in range(n_qubits):
#         qc.ry(params[i], i)
#     qc.cx(0,1); qc.cx(0,2); qc.cx(1,3); qc.cx(2,3)
#     for i in range(n_qubits):
#         qc.ry(params[n_qubits+i], i)
#     ansatz = qc.remove_final_measurements(inplace=False)

#     # Feature map
#     feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
#     qnn_circuit = feature_map.compose(ansatz, inplace=False)

#     # Estimator and gradient
#     estimator = Estimator()
#     gradient = ParamShiftEstimatorGradient(estimator)

#     # QNN
#     qnn = EstimatorQNN(
#         circuit=qnn_circuit,
#         input_params=feature_map.parameters,
#         weight_params=ansatz.parameters,
#         estimator=estimator,
#         gradient=gradient
#     )

#     # Callback function to monitor training progress
#     iteration = {"count": 0}  # mutable counter
#     def callback(xk):
#         iteration["count"] += 1
#         y_pred = qnn.forward(X_train_low, xk)
#         loss = np.mean((y_pred - y_train.ravel())**2)
#         print(f"Iteration {iteration['count']}, loss = {loss:.6f}")

#     # Optimizer with callback
#     optimizer = partial(
#         minimize,
#         method="L-BFGS-B",
#         options={"maxiter": maxiter},
#         callback=callback
#     )

#     # Train the regressor
#     start_time = time.time()
#     regressor = NeuralNetworkRegressor(neural_network=qnn, optimizer=optimizer)
#     regressor.fit(X_train_low, y_train.ravel())
#     train_time = time.time() - start_time

#     return regressor, pca, train_time


# import numpy as np
# from functools import partial
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
# from qiskit.circuit.library import ZZFeatureMap, TwoLocal
# from qiskit.primitives import Estimator
# from qiskit_machine_learning.neural_networks import EstimatorQNN
# from qiskit_machine_learning.gradients.param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
# from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
# from scipy.optimize import minimize
# import joblib
# import os

# def build_and_train_qnn(
#     X_train, y_train, y_scaler, 
#     n_qubits=2, reps=1, maxiter=100, batch=512, seed=42,
#     save_path="qnn_model.pkl"
# ):
#     rng = np.random.default_rng(seed)

#     # Use same target scaler as classical (already fitted in data_loader)
#     y_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

#     # Reduce dimensions
#     pca = PCA(n_components=n_qubits, random_state=seed)
#     X_low = pca.fit_transform(X_train)

#     # Scale inputs to angle range [0, 2π]
#     x_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
#     X_embed = x_scaler.fit_transform(X_low)

#     # Quantum circuit
#     feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
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
#     opt = partial(minimize, method="L-BFGS-B", options={"maxiter": maxiter})

#     regressor = NeuralNetworkRegressor(
#         neural_network=qnn,
#         optimizer=opt,
#         initial_point=w0,
#         loss="squared_error",
#     )
#     regressor.fit(X_embed, y_scaled)

#     pipeline = {
#         "model": regressor,
#         "pca": pca,
#         "x_scaler": x_scaler,
#         "y_scaler": y_scaler,  # shared scaler from data_loader
#     }

#     # Save pipeline
#     if save_path:
#         joblib.dump(pipeline, save_path)
#         print(f"[INFO] QNN pipeline saved to {save_path}")

#     return pipeline

# def qnn_predict(pipeline, X):
#     X_low = pipeline["pca"].transform(X)
#     X_embed = pipeline["x_scaler"].transform(X_low)
#     y_scaled = pipeline["model"].predict(X_embed).reshape(-1, 1)
#     return pipeline["y_scaler"].inverse_transform(y_scaled).ravel()

# def load_qnn_pipeline(path="qnn_model.pkl"):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"No saved QNN pipeline found at {path}")
#     pipeline = joblib.load(path)
#     print(f"[INFO] QNN pipeline loaded from {path}")
#     return pipeline


