import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Cargar y analizar los datos
data = pd.read_csv(r"C:/Users/beale/Desktop/DEEPFINAL/lechuza.csv")

# Análisis exploratorio
print(data.info())
print(data.describe())

# Manejo de valores nulos
data = data.dropna()

# Normalización de variables de entrada
scaler = StandardScaler()
X = scaler.fit_transform(data[['Radiacion', 'Temperatura', 'Temperatura panel']])
y = data['Potencia'].values

# División de datos
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Crear el modelo FFNN
def create_ffnn(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Capa de salida para regresión
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Crear el modelo
ffnn_model = create_ffnn((X_train.shape[1],))

# 3. Entrenamiento y evaluación
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamiento del modelo FFNN
history_ffnn = ffnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1
)

# Evaluación del modelo FFNN
ffnn_predictions = ffnn_model.predict(X_test)
ffnn_mse = mean_squared_error(y_test, ffnn_predictions)
ffnn_mae = mean_absolute_error(y_test, ffnn_predictions)
ffnn_r2 = r2_score(y_test, ffnn_predictions)

print("Resultados FFNN:")
print(f"MSE: {ffnn_mse}, MAE: {ffnn_mae}, R²: {ffnn_r2}")

# 4. Graficar el entrenamiento
def plot_training(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title(f'{model_name} - Pérdida Durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

plot_training(history_ffnn, "FFNN")

# Comparar las métricas
print(f"Resultados finales FFNN:")
print(f"- MSE: {ffnn_mse:.4f}")
print(f"- MAE: {ffnn_mae:.4f}")
print(f"- R²: {ffnn_r2:.4f}")
