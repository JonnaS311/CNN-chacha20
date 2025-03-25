import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

size = 50

def create_embedding_model(embedding_size=128):
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(size, size, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(embedding_size, activation=None),  # Embedding final sin activación
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # Normalización L2
    ])
    return model

'''
Triplet Loss : Usa un ancla, un positivo (misma persona) y
un negativo (otra persona), ajustando los embeddings para que la distancia 
entre ancla y positivo sea menor que con el negativo.
'''
def pairwise_distances(embeddings):
    """Calcula la matriz de distancias euclidianas entre embeddings."""
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)  # Evita valores negativos por errores numéricos
    return distances

def triplet_hard_loss(y_true, embeddings, margin=0.2):
    """Implementación manual de Triplet Hard Loss."""
    del y_true  # No se usa, solo es requerido por TensorFlow

    # Calcula la matriz de distancias entre embeddings
    distances = pairwise_distances(embeddings)

    # Obtiene los índices de anclas y positivos
    batch_size = tf.shape(embeddings)[0]
    labels = tf.range(batch_size)
    labels = tf.expand_dims(labels, 1)

    # Encuentra el positivo más cercano y el negativo más lejano
    mask_anchor_positive = tf.cast(tf.eye(batch_size), tf.bool)  # Máscara para evitar seleccionar la misma imagen
    hardest_positive_dist = tf.reduce_max(tf.where(mask_anchor_positive, tf.zeros_like(distances), distances), axis=1)

    mask_anchor_negative = tf.cast(~tf.eye(batch_size, dtype=tf.bool), tf.float32)  # Negativos que no sean el mismo
    hardest_negative_dist = tf.reduce_min(tf.where(mask_anchor_negative > 0, distances, tf.fill(tf.shape(distances), float("inf"))), axis=1)

    # Calcula la pérdida triplet loss con margen
    loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    return tf.reduce_mean(loss)

model = create_embedding_model()
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss=triplet_hard_loss)

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
history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)
model.save("modelo_entrenado.h5")

import matplotlib.pyplot as plt

# Obtener métricas del entrenamiento
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()