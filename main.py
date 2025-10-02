import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from auto_encoder import AutoEncoder


# Data preparation
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
val_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(latent_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 100
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


# Function to save representations
def save_representations(loader, split_name):
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


# Save representations for train and validation sets
save_representations(DataLoader(train_dataset, batch_size=256, shuffle=False), "train")
save_representations(DataLoader(val_dataset, batch_size=256, shuffle=False), "val")
# Visualize a random image and its reconstruction from the validation set


model.eval()
idx = random.randint(0, len(val_dataset) - 1)
img, _ = val_dataset[idx]
img_input = img.unsqueeze(0).to(device)
with torch.no_grad():
    recon, _ = model(img_input)
recon_img = recon.cpu().squeeze().numpy()
orig_img = img.squeeze().numpy()

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(orig_img, cmap="gray")
axs[0].set_title("Original")
axs[0].axis("off")
axs[1].imshow(recon_img, cmap="gray")
axs[1].set_title("Reconstruction")
axs[1].axis("off")
plt.tight_layout()
plt.savefig("reconstruction_example.png")
plt.close()

# Save the model
torch.save(model.state_dict(), "autoencoder_mnist.pth")
