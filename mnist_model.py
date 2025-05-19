
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Skapa en mapp för att spara modeller om den inte finns
os.makedirs("models", exist_ok=True)

# Ladda MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisera data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 1️⃣ Modell 1: Neuralt nätverk (Keras)
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print(f"Neural Network test accuracy: {acc:.4f}")
model.save("models/mnist_nn_model.keras")

# 2️⃣ Modell 2: Random Forest (scikit-learn)
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train_flat, y_train)
y_pred_rf = rf.predict(x_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest test accuracy: {acc_rf:.4f}")
joblib.dump(rf, "models/mnist_rf_model.pkl")
