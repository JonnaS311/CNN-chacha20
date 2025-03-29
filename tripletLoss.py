import tensorflow as tf

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

def mean_pos_distance(y_true, embeddings):
    # Calcula la matriz de distancias
    distances = pairwise_distances(embeddings)
    batch_size = tf.shape(embeddings)[0]
    # Usamos la máscara para ignorar la distancia consigo mismo (diagonal)
    mask_anchor_positive = tf.cast(tf.eye(batch_size), tf.bool)
    pos_distances = tf.where(mask_anchor_positive, tf.zeros_like(distances), distances)
    hardest_positive_dist = tf.reduce_max(pos_distances, axis=1)
    return tf.reduce_mean(hardest_positive_dist)

def mean_neg_distance(y_true, embeddings):
    distances = pairwise_distances(embeddings)
    batch_size = tf.shape(embeddings)[0]
    # Creamos una máscara para considerar todos los pares que no sean la misma imagen
    mask_anchor_negative = tf.cast(~tf.eye(batch_size, dtype=tf.bool), tf.float32)
    neg_distances = tf.where(mask_anchor_negative > 0, distances, tf.fill(tf.shape(distances), float("inf")))
    hardest_negative_dist = tf.reduce_min(neg_distances, axis=1)
    return tf.reduce_mean(hardest_negative_dist)


def l2_normalize_layer(x):
    import tensorflow as tf  # Importar tf dentro de la función
    return tf.math.l2_normalize(x, axis=1)


def triplet_accuracy(y_true, y_pred):
    # y_pred contiene los embeddings concatenados: [ancla, positivo, negativo]
    anchor = y_pred[:, :128]
    positive = y_pred[:, 128:256]
    negative = y_pred[:, 256:]
    
    # Calcular distancias
    dist_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    dist_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # Determinar si la distancia positiva es menor que la negativa + margen
    correct = tf.cast(dist_pos < dist_neg, tf.float32)
    return tf.reduce_mean(correct)