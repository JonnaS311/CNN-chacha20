import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras import regularizers
from tripletLoss import triplet_hard_loss, mean_neg_distance, mean_pos_distance, l2_normalize_layer, triplet_accuracy


size = 50 # tamaño en pixeles del modelo 
# Ruta a tu dataset
dataset_path = "./lfw-deepfunneled"  # Cambia esto a la ruta correcta
data_dir = pathlib.Path(dataset_path)

def create_embedding_model(embedding_size=128):

    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(size, size, 3)),
        layers.Dropout(0.3),  # Después del primer bloque convolucional
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Dropout(0.3),  # Después del primer bloque convolucional
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.Dropout(0.3),  # Después del primer bloque convolucional
        layers.GlobalAveragePooling2D(),
        layers.Dense(embedding_size, activation=None, kernel_regularizer=regularizers.l2(0.01)),  # Embedding final sin activación
        layers.Lambda(l2_normalize_layer)  # Normalización L2
    ])
    return model

def create_siamese_model(embedding_size=128):
    # Red base que comparten las tres entradas
    base_network = create_embedding_model(embedding_size)
    
    # Entradas para ancla, positivo y negativo
    input_anchor = layers.Input(shape=(size, size, 3), name='input_anchor')
    input_positive = layers.Input(shape=(size, size, 3), name='input_positive')
    input_negative = layers.Input(shape=(size, size, 3), name='input_negative')
    
    # Embeddings generados por la misma red
    embedding_anchor = base_network(input_anchor)
    embedding_positive = base_network(input_positive)
    embedding_negative = base_network(input_negative)
    
    # Concatenar embeddings para la métrica de accuracy
    concatenated = layers.Concatenate(axis=1, name='concat_embeddings')(
        [embedding_anchor, embedding_positive, embedding_negative]
    )
    
    return tf.keras.Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=concatenated
    )

model = create_siamese_model()

# Modelo auxiliar solo para extraer embeddings de anclas
embedding_model = tf.keras.Model(
    inputs=model.input[0],  # Toma solo el input_anchor
    outputs=model.layers[-2].output  # Capa antes de la concatenación
)

def generate_triplets(images, labels):
    batch_size = tf.shape(images)[0]
    
    def get_triplet(i):
        # Obtener etiqueta del ancla
        anchor_label = labels[i]
        
        # Encontrar índices positivos (misma clase)
        positive_mask = tf.equal(labels, anchor_label)
        positive_indices = tf.where(positive_mask)
        positive_idx = tf.random.shuffle(positive_indices)[0]
        positive_idx = tf.gather(positive_idx, 0)  # Convertir a escalar
        
        # Encontrar índices negativos (clase diferente)
        negative_mask = tf.logical_not(positive_mask)
        negative_indices = tf.where(negative_mask)
        negative_idx = tf.random.shuffle(negative_indices)[0]
        negative_idx = tf.gather(negative_idx, 0)  # Convertir a escalar
        
        return (
            images[i], 
            images[positive_idx], 
            images[negative_idx]
        )
    
    # Generar tripletas para todo el batch
    indices = tf.range(batch_size)
    anchors, positives, negatives = tf.map_fn(
        get_triplet, 
        indices, 
        fn_output_signature=(tf.float32, tf.float32, tf.float32)
    )
    
    return (anchors, positives, negatives)

# Modifica la función map_to_triplets para retener las etiquetas originales
def map_to_triplets(image_batch, label_batch):
    anchors, positives, negatives = generate_triplets(image_batch, label_batch)
    return (anchors, positives, negatives), label_batch  # ¡Ahora retornamos las etiquetas reales!

# Crear dataset de entrenamiento y validación (80% train, 20% val)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(size, size),
    batch_size=128
).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(size, size),
    batch_size=128
).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))  # Normalización aquí

train_dataset = train_dataset.map(map_to_triplets)
val_dataset = val_dataset.map(map_to_triplets)


def custom_triplet_loss(y_true, y_pred):
    return triplet_hard_loss(y_true, y_pred[:, :128])  # Usa la misma lógica que la lambda

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=custom_triplet_loss, 
    metrics=[triplet_accuracy, mean_pos_distance, mean_neg_distance]
)

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    # Asegurar que los datos se pasen como tripletas
    )




# Extraer embeddings y etiquetas del conjunto de validación
embeddings_val = []
y_val = []

for batch in val_dataset.take(1):  # Toma 1 batch para visualización
    (anchors, _, _), labels = batch
    embeddings = embedding_model.predict(anchors)
    embeddings_val.extend(embeddings)
    y_val.extend(labels.numpy())

embeddings_val = np.array(embeddings_val)
y_val = np.array(y_val)

# Reducción de dimensionalidad con TSNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_val)

# Visualización
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_val, cmap='gist_ncar', alpha=0.6)
plt.colorbar()
plt.title("Visualización de Embeddings (TSNE)")
plt.savefig('embeddings_visualization.png')
plt.close()

model.save("modelo_entrenado.keras")

# Obtener métricas del entrenamiento
plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 4, 1)
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title("Pérdida")

# Distancia promedio ancla-positivo
plt.subplot(1, 4, 2)
plt.plot(history.history['mean_pos_distance'], label='Distancia Ancla-Positivo')
plt.plot(history.history['val_mean_pos_distance'], label='Distancia Val Ancla-Positivo')
plt.xlabel('Épocas')
plt.ylabel('Distancia')
plt.legend()
plt.title("Distancia Ancla-Positivo")

# Distancia promedio ancla-negativo
plt.subplot(1, 4, 3)
plt.plot(history.history['mean_neg_distance'], label='Distancia Ancla-Negativo')
plt.plot(history.history['val_mean_neg_distance'], label='Distancia Val Ancla-Negativo')
plt.xlabel('Épocas')
plt.ylabel('Distancia')
plt.legend()
plt.title("Distancia Ancla-Negativo")

plt.subplot(1, 4, 4)
plt.plot(history.history['triplet_accuracy'], label='Accuracy Entrenamiento')
plt.plot(history.history['val_triplet_accuracy'], label='Accuracy Validación')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
#plt.show()
plt.savefig('mi_grafico.png') 