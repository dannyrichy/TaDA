import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np

import torch.nn as nn
import torch.optim as optim

# Autoencoder definition
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Data preparation
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder(latent_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
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
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")

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
    os.makedirs('representations', exist_ok=True)
    for label, reps in reps_by_label.items():
        reps = np.stack(reps)
        np.save(f'representations/{split_name}_label_{label}.npy', reps)

# Save representations for train and validation sets
save_representations(DataLoader(train_dataset, batch_size=256, shuffle=False), 'train')
save_representations(DataLoader(val_dataset, batch_size=256, shuffle=False), 'val')