import kagglehub
import pandas as pd  
import os
import shutil


min_images = 100  # Cambia este valor según lo que necesites


try:
    shutil.rmtree(r"C:\Users\jhony\.cache\kagglehub\datasets")  # Elimina la carpeta y su contenido
except:
   print("no existe ese archivo")

path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

print("Path to dataset files:", path)


df = pd.read_csv(f"{path}/lfw_allnames.csv")  
filtrado = df[df["images"] >= min_images]  
print(filtrado)


dataset_dir = os.path.join(path,"lfw-deepfunneled","lfw-deepfunneled")
for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    
    if os.path.isdir(person_dir):  # Verifica que es una carpeta
        num_images = len(os.listdir(person_dir))
        if num_images < min_images:
            shutil.rmtree(person_dir)  # Elimina la carpeta completa

print("Filtrado completado.")

shutil.move(dataset_dir, os.getcwd())  # Mueve la carpeta al nuevo destino

print("carpeta recuperada")