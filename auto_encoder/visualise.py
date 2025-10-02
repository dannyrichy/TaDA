import torch
import random

import matplotlib.pyplot as plt
from auto_encoder.model import AutoEncoder
from torchvision.datasets import MNIST
from torchvision import transforms


def visualise(
    model_path="autoencoder_mnist.pth", dataset_root="./data"
):
    val_dataset = MNIST(
        root=dataset_root, train=False, download=True, transform=transforms.ToTensor()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    plt.savefig(f"reconstruction_example_{random.randint(0, 99999)}.png")
    plt.close()
