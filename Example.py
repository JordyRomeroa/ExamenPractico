import tensorflow as tf

print("GPUs físicas:", tf.config.list_physical_devices('GPU'))

# Modelo sencillo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

import numpy as np
x = np.random.randn(5000, 100).astype('float32')
y = np.random.randint(0, 10, size=(5000,))

# Entrenamiento
model.fit(x, y, epochs=5, batch_size=128)

# Ver dispositivos lógicos GPU
print("GPUs lógicas:", tf.config.list_logical_devices('GPU'))
