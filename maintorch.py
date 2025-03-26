import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # Librería para mostrar barras de progreso

# Parámetros
size = 60  # tamaño de la imagen (60x60)
embedding_size = 128
batch_size = 64
num_epochs = 10
margin = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo de embedding similar al de Keras
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)  # asume imágenes de 3 canales
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(256, embedding_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Normalización L2
        x = F.normalize(x, p=2, dim=1)
        return x

# Instanciamos el modelo y lo mandamos a dispositivo
model = EmbeddingNet(embedding_size=embedding_size).to(device)

# Usamos la implementación nativa de Triplet Loss de PyTorch
triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Transformaciones para las imágenes (incluye normalización)
data_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    # Normalización, ajusta según tu dataset:
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Ejemplo: Usando ImageFolder para cargar un dataset organizado en carpetas
# Debes crear o adaptar un Dataset que genere tripletas.
dataset_path = "./lfw-deepfunneled"  # Cambia a la ruta de tu dataset

# Aquí usamos ImageFolder; sin embargo, esto retorna (imagen, label) y no tripletas.
# Se recomienda implementar un dataset que forme tripletas. 
# A modo de ejemplo, se define una clase DummyTripletDataset que genera tripletas a partir de ImageFolder.
class DummyTripletDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.transform = transform
        # Organiza índices por clase para facilitar la creación de tripletas.
        self.class_to_idx = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_to_idx.setdefault(label, []).append(idx)
    
    def __getitem__(self, index):
        # Obtenemos la imagen ancla y su etiqueta
        anchor_img, anchor_label = self.dataset[index]
        # Seleccionamos un positivo distinto
        positive_idx = index
        while positive_idx == index:
            positive_idx = np.random.choice(self.class_to_idx[anchor_label])
        positive_img, _ = self.dataset[positive_idx]
        # Seleccionamos un negativo de otra clase
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = np.random.choice(list(self.class_to_idx.keys()))
        negative_idx = np.random.choice(self.class_to_idx[negative_label])
        negative_img, _ = self.dataset[negative_idx]
        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset)

# Creamos los datasets de entrenamiento y validación
# Nota: aquí se usa el mismo dataset para ambos; en la práctica, separa según corresponda.
train_dataset = DummyTripletDataset(root=dataset_path, transform=data_transforms)
val_dataset = DummyTripletDataset(root=dataset_path, transform=data_transforms)

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Para almacenar métricas por época
    train_losses, val_losses = [], []
    train_pos_dists, train_neg_dists, train_triplet_accs = [], [], []
    val_pos_dists, val_neg_dists, val_triplet_accs = [], [], []
    # Bucle de entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_pos_dist = 0.0
        running_neg_dist = 0.0
        running_acc = 0.0

        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for anchor, positive, negative in train_loader_iter:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)
            
            loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            # Distancias
            pos_dist = F.pairwise_distance(anchor_out, positive_out).mean().item()
            neg_dist = F.pairwise_distance(anchor_out, negative_out).mean().item()
            
            # Precisión de tripletas (si el negativo está más lejos que el positivo)
            correct = (neg_dist > pos_dist + margin)
            acc = correct * 100  # en porcentaje

            # Acumulamos valores para calcular promedios
            running_loss += loss.item() * anchor.size(0)
            running_pos_dist += pos_dist * anchor.size(0)
            running_neg_dist += neg_dist * anchor.size(0)
            running_acc += acc * anchor.size(0)

            train_loader_iter.set_postfix(loss=loss.item(), pos_dist=pos_dist, neg_dist=neg_dist, acc=acc)

        # Promediamos las métricas de entrenamiento
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_pos_dist = running_pos_dist / len(train_loader.dataset)
        epoch_neg_dist = running_neg_dist / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_pos_dists.append(epoch_pos_dist)
        train_neg_dists.append(epoch_neg_dist)
        train_triplet_accs.append(epoch_acc)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, PosDist={epoch_pos_dist:.4f}, NegDist={epoch_neg_dist:.4f}, TripletAcc={epoch_acc:.2f}%")

        # Validación
        model.eval()
        running_val_loss, running_val_pos_dist, running_val_neg_dist, running_val_acc = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)
                
                loss = triplet_loss_fn(anchor_out, positive_out, negative_out)

                pos_dist = F.pairwise_distance(anchor_out, positive_out).mean().item()
                neg_dist = F.pairwise_distance(anchor_out, negative_out).mean().item()
                correct = (neg_dist > pos_dist + margin)
                acc = correct * 100

                running_val_loss += loss.item() * anchor.size(0)
                running_val_pos_dist += pos_dist * anchor.size(0)
                running_val_neg_dist += neg_dist * anchor.size(0)
                running_val_acc += acc * anchor.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_pos_dist = running_val_pos_dist / len(val_loader.dataset)
        epoch_val_neg_dist = running_val_neg_dist / len(val_loader.dataset)
        epoch_val_acc = running_val_acc / len(val_loader.dataset)

        val_losses.append(epoch_val_loss)
        val_pos_dists.append(epoch_val_pos_dist)
        val_neg_dists.append(epoch_val_neg_dist)
        val_triplet_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}: Val Loss={epoch_val_loss:.4f}, PosDist={epoch_val_pos_dist:.4f}, NegDist={epoch_val_neg_dist:.4f}, TripletAcc={epoch_val_acc:.2f}%")
        
        
    # Guardar el modelo
    torch.save(model.state_dict(), "modelo_entrenado.pt")

    # Graficar la evolución de la pérdida
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Loss Entrenamiento')
    plt.plot(val_losses, label='Loss Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title("Pérdida de Triplet Loss")
    plt.legend()
    plt.show()