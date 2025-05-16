import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from unet import UNet

# --- Carga y preprocesamiento de imágenes y máscaras ---
image_dir = './Actividad_2/data/Image/'
mask_dir = './Actividad_2/data/Mask/'

images = os.listdir(image_dir)
image_tensor = []
masks_tensor = []

for image_name in images:
    image_path = os.path.join(image_dir, image_name)
    img = Image.open(image_path)
    img_tensor = torchvision.transforms.functional.pil_to_tensor(img)
    img_tensor = torchvision.transforms.functional.resize(img_tensor, (100, 100))
    img_tensor = img_tensor[None, :, :, :]
    img_tensor = torch.tensor(img_tensor, dtype=torch.float) / 255.

    if img_tensor.shape != (1, 3, 100, 100):
        continue

    # Máscara correspondiente
    mask_name = image_name.replace('.jpg', '.png')
    mask_path = os.path.join(mask_dir, mask_name)
    mask_img = Image.open(mask_path)
    mask_tensor = torchvision.transforms.functional.pil_to_tensor(mask_img)
    mask_tensor = torchvision.transforms.functional.resize(mask_tensor, (100, 100))
    mask_tensor = mask_tensor[:1, :, :]  # Usar solo un canal
    mask_tensor = (mask_tensor > 0).long().squeeze(0)  # shape (100, 100) con valores 0 o 1

    image_tensor.append(img_tensor)
    masks_tensor.append(mask_tensor)

# --- División en train/test ---
image_tensor = torch.cat(image_tensor)
masks_tensor = torch.stack(masks_tensor)

indices = list(range(len(image_tensor)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_images = image_tensor[train_idx]
train_masks = masks_tensor[train_idx]
val_images = image_tensor[val_idx]
val_masks = masks_tensor[val_idx]

train_loader = torch.utils.data.DataLoader(list(zip(train_images, train_masks)), batch_size=64)
val_loader = torch.utils.data.DataLoader(list(zip(val_images, val_masks)), batch_size=64)

# --- Configuración del modelo ---
model = UNet(n_channels=3, n_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Entrenamiento y evaluación ---
num_epochs = 20
train_loss_list, val_loss_list = [], []
train_jaccard_list, val_jaccard_list = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    jaccard_epoch = []

    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, pred_flat = torch.max(pred, 1)
        y_flat = y
        intersection = torch.sum(pred_flat == y_flat, dim=(1, 2)) / 10000.0
        jaccard_epoch.append(torch.mean(intersection).detach())

    train_loss = running_loss
    train_jaccard = sum(jaccard_epoch) / len(jaccard_epoch)
    train_loss_list.append(train_loss)
    train_jaccard_list.append(train_jaccard)

    # --- Evaluación ---
    model.eval()
    val_loss = 0.0
    val_jaccard_epoch = []

    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

            _, pred_flat = torch.max(pred, 1)
            y_flat = y
            intersection = torch.sum(pred_flat == y_flat, dim=(1, 2)) / 10000.0
            val_jaccard_epoch.append(torch.mean(intersection).detach())

    val_loss_list.append(val_loss)
    val_jaccard = sum(val_jaccard_epoch) / len(val_jaccard_epoch)
    val_jaccard_list.append(val_jaccard)

    # --- Imprimir métricas ---
    print(f"Época [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
          f"Train Jaccard: {train_jaccard:.4f}, Val Jaccard: {val_jaccard:.4f}")

# --- Guardar gráficas ---
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title("Pérdida por época")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")

plt.figure(figsize=(10, 5))
plt.plot(train_jaccard_list, label='Train Jaccard')
plt.plot(val_jaccard_list, label='Validation Jaccard')
plt.title("Jaccard Index por época")
plt.xlabel("Épocas")
plt.ylabel("Jaccard Index")
plt.legend()
plt.grid(True)
plt.savefig("jaccard_plot.png")
