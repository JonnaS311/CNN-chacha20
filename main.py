import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models

def create_embedding_model(embedding_size=128):
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(160, 160, 3)),
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
Triplet Loss (usada en FaceNet): Usa un ancla, un positivo (misma persona) y
un negativo (otra persona), ajustando los embeddings para que la distancia 
entre ancla y positivo sea menor que con el negativo.
'''
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)


model = create_embedding_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=triplet_loss)

# Entrenamiento con tripletas de imágenes
model.fit(train_dataset, epochs=50, validation_data=val_dataset)