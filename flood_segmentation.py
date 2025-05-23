import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unet import UNet

# --- Confiuración ---
num_epochs= 30
lr = 1e-4
test_size = 0.2
batch_size = 16

# --- Dataset personalizado ---
class FloodDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Imagen RGB garantizada
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = TF.pil_to_tensor(image).float() / 255.0  # [3, H, W]
        image = TF.resize(image, (100, 100))

        # Máscara monocanal
        mask = Image.open(self.mask_paths[idx])
        mask = TF.pil_to_tensor(mask)[0].unsqueeze(0)  # [1, H, W]
        mask = TF.resize(mask, (100, 100))
        mask = mask.squeeze(0)                         # [H, W]
        mask = (mask > 0).float().unsqueeze(0)  # [1, H, W] binaria y float

        return image, mask


# --- Carga de rutas de imágenes y máscaras ---
image_dir = 'data/Image/'
mask_dir = 'data/Mask/'

image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files = sorted([os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in os.listdir(image_dir) if f.endswith('.jpg')])

# --- División en entrenamiento y validación ---
train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=test_size, random_state=42)

train_dataset = FloodDataset(train_imgs, train_masks)
val_dataset = FloodDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Modelo, pérdida, optimizador ---
model = UNet(n_channels=3, n_classes=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Entrenamiento ---
num_epochs = num_epochs
train_loss_list, val_loss_list = [], []
train_jaccard_list, val_jaccard_list = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, train_jaccard = 0.0, []

    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)  # [B, 2, H, W]
        loss = criterion(pred, y)  # pred: [B, 1, H, W], y: [B, 1, H, W]
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred_classes = (torch.sigmoid(pred) > 0.5).float()  # binariza
        jaccard = torch.mean((pred_classes == y).float(), dim=(1, 2, 3))
        train_jaccard.append(jaccard.mean().item())

    train_loss /= len(train_loader)
    train_jaccard_mean = sum(train_jaccard) / len(train_jaccard)
    train_loss_list.append(train_loss)
    train_jaccard_list.append(train_jaccard_mean)

    model.eval()
    val_loss, val_jaccard = 0.0, []

    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

            pred_classes = torch.argmax(pred, dim=1)
            jaccard = torch.mean((pred_classes == y).float(), dim=(1, 2))
            val_jaccard.append(jaccard.mean().item())

    val_loss /= len(val_loader)
    val_jaccard_mean = sum(val_jaccard) / len(val_jaccard)
    val_loss_list.append(val_loss)
    val_jaccard_list.append(val_jaccard_mean)

    print(f"Época [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
          f"Train Jaccard: {train_jaccard_mean:.4f}, Val Jaccard: {val_jaccard_mean:.4f}")

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
