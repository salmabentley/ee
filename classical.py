
# -----------------------
# cnn
# -----------------------
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

input_layer = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear')(x)
classical_model = Model(inputs=input_layer, outputs=output)

classical_model.compile(optimizer=Adam(learning_rate=1e-3),
                        loss='mse', metrics=['mae'])

classical_model.fit(X_train, y_train, validation_split=0.2,
                    epochs=50, batch_size=32, verbose=1)

y_pred_classical = classical_model.predict(X_test)
y_pred_classical_rescaled = y_scaler.inverse_transform(y_pred_classical)