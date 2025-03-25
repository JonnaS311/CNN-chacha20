import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

print(tf.test.is_built_with_cuda())  # Debe devolver True
print(tf.config.list_physical_devices('GPU'))  # Lista GPUs disponibles

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))