from tensorflow import keras
import time
import joblib
import os

def build_and_train(X_train, y_train, epochs=10, batch_size=32, save_path="cls_model.pkl"):
    input_layer = keras.Input(shape=(X_train.shape[1],))
    x = keras.layers.Dense(64, activation="relu")(input_layer)
    x = keras.layers.Dense(32, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    train_time = time.time() - start_time

    # Save the model
    if save_path:
        # Use modern Keras format instead of h5
        model.save("cls_model.keras")  
        joblib.dump({"history": history.history, "train_time": train_time}, save_path)
        print(f"[INFO] Classical model saved to cls_model.keras and {save_path}")


    return model, history, train_time


def load_classical(save_path="cls_model.pkl"):
    if not os.path.exists("cls_model.keras") or not os.path.exists(save_path):
        raise FileNotFoundError("[ERROR] No saved classical model found")

    model = keras.models.load_model("cls_model.keras")
    meta = joblib.load(save_path)
    print(f"[INFO] Classical model loaded from cls_model.keras and {save_path}")
    return model, meta["history"], meta["train_time"]

