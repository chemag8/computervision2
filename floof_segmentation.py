import torch

import torch.nn as nn # para definir la red neuronal
import torch.optim as optim # para optimizar la red neuronal
import torchvision # para usar torchvision.transforms.functional

import os
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import unet


if __name__ == '__main__':
    images = os.listdir('data/Image/')
    masks = os.listdir('data/Mask/')

    print(len(images), len(masks))
   
    image_tensor = list()
    masks_tensor = list()

    for image in images:
        dd = Image.open(f'data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))

        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.

        if tt.shape != (1, 3, 100, 100):
            continue

        image_tensor.append(tt)
        
	mask = image.replace('.jpg', '.png')
	dd = Image.open(f'data/Mask/{mask}')
	mm = torchvision.transforms.functional.pil_to_tensor(dd)
	mm = mm.repeat(3, 1, 1)
	mm = torchvision.transforms.functional.resize(mm, (100, 100))
	mm = mm[:1, :, :]  # tomamos solo 1 canal

	# Binarizamos: todo píxel > 0 se convierte en clase 1
	mm = torch.tensor((mm > 0).detach().numpy(), dtype=torch.long)

	# One-hot encoding: convertimos cada valor a vector clase
	mm = torch.nn.functional.one_hot(mm)

	# Rearmamos las dimensiones para PyTorch: (C, H, W)
	mm = torch.permute(mm, (0, 3, 1, 2))
	mm = torch.tensor(mm, dtype=torch.float)

	masks_tensor.append(mm)
    
    # Concatenamos todas las imágenes y máscaras procesadas en tensores
    image_tensor = torch.cat(image_tensor)  # [N, 3, H, W]
    print(image_tensor.shape)

    masks_tensor = torch.cat(masks_tensor)  # [N, 2, H, W] si hiciste one-hot
    print(masks_tensor.shape)

    # Definimos el modelo U-Net
    model_unet = unet.UNet(n_channels=3, n_classes=2)

    # Creamos los dataloaders para imágenes y máscaras por separado (no recomendado)
    dataloader_train_image = torch.utils.data.DataLoader(image_tensor, batch_size=64)
    dataloader_train_target = torch.utils.data.DataLoader(masks_tensor, batch_size=64)

    # Optimizador
    optim = torch.optim.Adam(model_unet.parameters(), lr=1e-3)

    # Función de pérdida para segmentación multiclase (aunque tengas 2 clases)
    cross_entropy = torch.nn.CrossEntropyLoss()

    loss_list = list()
    jaccard_list = list()

    for epoch in range(20):
        running_loss = 0.0
        model_unet.train()

        jaccard_epoch = list()

        for image, target in zip(dataloader_train_image, dataloader_train_target):
            pred = model_unet(image)  # Predicción de la U-Net

            loss = cross_entropy(pred, target)  # Calculamos pérdida
            running_loss += loss.item()

            loss.backward()
            optim.step()
            
        for image, target in zip(dataloader_train_image, dataloader_train_target):
            model_unet.eval()
            pred = model_unet(image)

            _, pred_unflatten = torch.max(pred, dim=1)
            _, target_unflatten = torch.max(target, dim=1)

            # Cálculo de intersección (número de píxeles iguales entre predicción y ground truth)
            intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1, 2)) / 10000.0

            # Promedio de precisión por batch
            jaccard_epoch.append(torch.mean(intersection).detach())
        
        jaccard_list.append(sum(jaccard_epoch)/len(jaccard_epoch))
        loss_list.append(running_loss)
        
        print(f"Epoch {epoch+1}/{10}, Loss: {running_loss:.4f}, Jaccard: {jaccard_list[-1]:.4f}")
        
    # Graficamos la pérdida y el índice de Jaccard
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 filas, 1 columna

    # Gráfico de pérdida
    axs[0].plot(loss_list, label='Loss', color='tab:blue')
    axs[0].set_title('Evolución de la Pérdida')
    axs[0].set_xlabel('Épocas')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()

    # Gráfico de Jaccard
    axs[1].plot(jaccard_list, label='Jaccard Index', color='tab:green')
    axs[1].set_title('Evolución del Jaccard Index')
    axs[1].set_xlabel('Épocas')
    axs[1].set_ylabel('IoU')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


