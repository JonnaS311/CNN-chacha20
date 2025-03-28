import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import matplotlib.pyplot as plt
from tripletLoss import triplet_hard_loss, mean_neg_distance, mean_pos_distance, l2_normalize_layer

size = 60# tamaño en pixeles del modelo 



def create_embedding_model(embedding_size=128):

    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(size, size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(embedding_size, activation=None),  # Embedding final sin activación
        layers.Lambda(l2_normalize_layer)  # Normalización L2
    ])
    return model

model = create_embedding_model()
model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), 
              loss=triplet_hard_loss,
              metrics=[mean_pos_distance, mean_neg_distance])

# Ruta a tu dataset
dataset_path = "./lfw-deepfunneled"  # Cambia esto a la ruta correcta
data_dir = pathlib.Path(dataset_path)

# Crear dataset de entrenamiento y validación (80% train, 20% val)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 20% para validación
    subset="training",
    seed=123,  # Asegura consistencia en la división
    image_size=(size, size),  # Ajusta según la red que uses
    batch_size=64
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(size, size),
    batch_size=64
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Entrenamiento con tripletas de imágenes
history = model.fit(train_dataset, epochs=3, validation_data=val_dataset)
model.save("modelo_entrenado.h5")


# Obtener métricas del entrenamiento
plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title("Pérdida")

# Distancia promedio ancla-positivo
plt.subplot(1, 3, 2)
plt.plot(history.history['mean_pos_distance'], label='Distancia Ancla-Positivo')
plt.plot(history.history['val_mean_pos_distance'], label='Distancia Val Ancla-Positivo')
plt.xlabel('Épocas')
plt.ylabel('Distancia')
plt.legend()
plt.title("Distancia Ancla-Positivo")

# Distancia promedio ancla-negativo
plt.subplot(1, 3, 3)
plt.plot(history.history['mean_neg_distance'], label='Distancia Ancla-Negativo')
plt.plot(history.history['val_mean_neg_distance'], label='Distancia Val Ancla-Negativo')
plt.xlabel('Épocas')
plt.ylabel('Distancia')
plt.legend()
plt.title("Distancia Ancla-Negativo")

plt.tight_layout()
#plt.show()
plt.savefig('mi_grafico.png') 