# from config import *
# from data_loader import load_and_preprocess
# from classical import build_and_train
# from quantum import build_and_train_qnn
# from evaluations import evaluate
# from visualisations import plot_predictions

# # Load data
# X_train, X_test, y_train, y_test, y_scaler = load_and_preprocess()

# # Quantum
# qnn_regressor, pca, time_qnn, weights = build_and_train_qnn(X_train, y_train)

# # Classical
# cls_model, history, time_cls = build_and_train(X_train, y_train)


# # Predictions
# X_test_low = pca.transform(X_test)
# y_pred_qnn = qnn_regressor.predict(X_test_low)
# y_pred_qnn_rescaled = y_scaler.inverse_transform(y_pred_qnn.reshape(-1,1))

# y_pred_cls = cls_model.predict(X_test)
# y_pred_cls_rescaled = y_scaler.inverse_transform(y_pred_cls)
# y_test_rescaled = y_scaler.inverse_transform(y_test)


# # Evaluation
# metrics_qnn = evaluate(y_test_rescaled, y_pred_qnn_rescaled)
# metrics_cls = evaluate(y_test_rescaled, y_pred_cls_rescaled)

# print("Quantum NN metrics:", metrics_qnn)
# print("Classical NN metrics:", metrics_cls)

# # Visualisation
# plot_predictions(y_test_rescaled.flatten(), 
#                  y_pred_cls_rescaled.flatten(), 
#                  y_pred_qnn_rescaled.flatten())
import os
import joblib
from config import *
from data_loader import load_and_preprocess
from classical import build_and_train, load_classical
from quantum import build_and_train_qnn, qnn_predict, load_qnn_pipeline
from evaluations import evaluate
from visualisations import plot_predictions


# =============================
# Load data
# =============================
X_train, X_test, y_train, y_test, y_scaler = load_and_preprocess()


# =============================
# Quantum Model (train or load)
# =============================
# QNN_MODEL_PATH = "qnn_model.pkl"

# if os.path.exists(QNN_MODEL_PATH):
#     # Forcing a rebuild for debugging purposes
#     print("[INFO] Old QNN model exists. Deleting and retraining...")
#     os.remove(QNN_MODEL_PATH)
    
# qnn_pipeline = build_and_train_qnn(
#     X_train[:500], y_train[:500],
#     reps=3, maxiter=10,
# )

# y_pred_qnn_rescaled = qnn_predict(qnn_pipeline, X_test)


# =============================
# Classical Model (train or load)
# =============================
CLS_MODEL_PATH = "cls_model.pkl"

from classical import build_and_train, load_classical

if os.path.exists("cls_model.keras") and os.path.exists("cls_model.pkl"):
    cls_model, history, time_cls = load_classical()
else:
    cls_model, history, time_cls = build_and_train(X_train, y_train, save_path="cls_model.pkl")


y_pred_cls = cls_model.predict(X_test)
y_pred_cls_rescaled = y_scaler.inverse_transform(y_pred_cls)
y_test_rescaled = y_scaler.inverse_transform(y_test)


# =============================
# Evaluation
# =============================
metrics_qnn = evaluate(y_test_rescaled, y_pred_qnn_rescaled)
metrics_cls = evaluate(y_test_rescaled, y_pred_cls_rescaled)

print("Quantum NN metrics:", metrics_qnn)
print("Classical NN metrics:", metrics_cls)


# =============================
# Visualisation
# =============================
plot_predictions(
    y_test_rescaled.flatten(),
    y_pred_cls_rescaled.flatten(),
    y_pred_qnn_rescaled.flatten(),
)

