import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_cls, y_qnn):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_cls, alpha=0.5, label="Classical NN", s=20)
    plt.scatter(y_true, y_qnn, alpha=0.5, label="Quantum NN", s=20)
    mn, mx = np.min(y_true), np.max(y_true)
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("True pt1")
    plt.ylabel("Predicted pt1")
    plt.legend()
    plt.show()
