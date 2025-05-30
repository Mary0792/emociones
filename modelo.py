import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Cargar el dataset FER-2013
data = pd.read_csv('fer2013.csv')

# Procesamiento de datos
def preprocess(data):
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pix, sep=' ') for pix in pixels], dtype='float32')
    images = images.reshape((-1, 48, 48, 1)) / 255.0  # Normalizamos
    labels = to_categorical(data['emotion'], num_classes=7)
    return images, labels

X, y = preprocess(data)

# Emociones del dataset (puedes reducirlo a 5)
emotion_labels = {
    0: "Enojado",
    1: "Disgusto",     # podrías omitir esta
    2: "Miedo",
    3: "Feliz",
    4: "Triste",
    5: "Sorpresa",
    6: "Neutral"
}

# Si deseas solo 5 emociones, filtra así:
selected_classes = [0, 2, 3, 4, 6]  # Enojado, Miedo, Feliz, Triste, Neutral
idx = [i for i, label in enumerate(np.argmax(y, axis=1)) if label in selected_classes]
X = X[idx]
y = y[idx]
y = to_categorical(np.argmax(y, axis=1), num_classes=len(selected_classes))

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(selected_classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=64)

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión en datos de prueba: {acc*100:.2f}%")
