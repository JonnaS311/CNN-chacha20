import tensorflow as tf
from tripletLoss import triplet_hard_loss, mean_neg_distance, mean_pos_distance, l2_normalize_layer


# Cargar el modelo guardado (ajusta las funciones personalizadas)
model = tf.keras.models.load_model(
    "modelo_entrenado.h5",
    custom_objects={
        "triplet_hard_loss": triplet_hard_loss,
        "mean_pos_distance": mean_pos_distance,
        "mean_neg_distance": mean_neg_distance,
        "l2_normalize_layer": l2_normalize_layer
    }
)

import numpy as np
import os
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, target_size=(92, 92)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    return np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

# Ruta a la carpeta de referencia
reference_dir = "./lfw-deepfunneled"

# Diccionario para almacenar {nombre: lista de embeddings}
reference_embeddings = {}

for person_name in os.listdir(reference_dir):
    person_dir = os.path.join(reference_dir, person_name)
    embeddings = []
    
    count = 0
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = preprocess_image(img_path)
        embedding = model.predict(img)[0]  # Shape: (128,)
        embeddings.append(embedding)
        print(embedding)
        print(person_name)
        count += 1
        if count == 3:
            break
    # Almacenar el promedio de los embeddings de la persona (opcional)
    reference_embeddings[person_name] = np.mean(embeddings, axis=0)

np.save("referencia_embeddings.npy", reference_embeddings)

print(reference_embeddings)