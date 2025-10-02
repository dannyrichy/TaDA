import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from auto_encoder.model import AutoEncoder


def train_autoencoder(
    train_loader,
    val_loader,
    latent_dim=32,
    batch_size=128,
    epochs=100,
    lr=1e-3,
    model_save_path="autoencoder_mnist.pth",
    device=None,
):

    # Model, loss, optimizer
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                val_loss += loss.item() * imgs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
    # Save the model
    torch.save(model.state_dict(), model_save_path)
    return model


# Function to save representations
def save_representations(model, loader, device, split_name):
    model.eval()
    reps_by_label = {i: [] for i in range(10)}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, z = model(imgs)
            z = z.cpu().numpy()
            for zi, label in zip(z, labels):
                reps_by_label[label.item()].append(zi)
    os.makedirs("representations", exist_ok=True)
    for label, reps in reps_by_label.items():
        reps = np.stack(reps)
        np.save(f"representations/{split_name}_label_{label}.npy", reps)


if __name__ == "__main__":
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    # Data preparation
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_autoencoder(train_loader=train_loader, val_loader=val_loader)

    save_representations(model, train_loader, device, "train")
    save_representations(model, val_loader, device, "val")