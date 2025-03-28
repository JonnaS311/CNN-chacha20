import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tripletLoss import triplet_hard_loss, mean_neg_distance, mean_pos_distance, l2_normalize_layer

def preprocess_image(image_path, target_size=(92, 92)):
    img = image.load_img(image_path, target_size=target_size)  # Redimensionar
    img_array = image.img_to_array(img)  # Convertir a array (shape: (92, 92, 3))
    img_array = img_array / 255.0  # Normalizar (como en Rescaling(1./255)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch (shape: (1, 92, 92, 3))
    return img_array

# Ejemplo: Cargar una imagen
new_image = preprocess_image("cara.jpg")


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

new_embedding = model.predict(new_image)  # Shape: (1, 128)

reference_embeddings = np.load("referencia_embeddings.npy", allow_pickle=True).item()

# Calcular distancias (ej. distancia coseno)
def cosine_distance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

distances = {}
for name, ref_embedding in reference_embeddings.items():
    distances[name] = np.linalg.norm(new_embedding - ref_embedding)  # Distancia euclidiana

# Paso 5: Decidir
predicted_name = min(distances, key=distances.get)
if distances[predicted_name] < 0.5:
    print(f"Identidad: {predicted_name}")
else:
    print("Desconocido")