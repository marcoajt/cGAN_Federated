"""Gera dados sintéticos com labels usando o gerador salvo."""

import argparse
import torch
import matplotlib.pyplot as plt
from gan_federated.task import Generator


def generate_synthetic(class_label: int, num_samples: int = 10, model_path: str = "generator_client0.pt"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    n_classes = 10
    img_shape = (1, 28, 28)
    # Carrega o gerador
    gen = Generator(latent_dim, n_classes, img_shape).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    # Amostras sintéticas
    noise = torch.randn(num_samples, latent_dim, device=device)
    labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
    with torch.no_grad():
        imgs = gen(noise, labels).cpu()
    # Desfaz normalização de [-1,1] para [0,1]
    imgs = (imgs + 1) / 2
    return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera imagens sintéticas por classe.")
    parser.add_argument("--class_label", type=int, default=5, help="Rótulo da classe (0-9)")
    parser.add_argument("--num_samples", type=int, default=10, help="Número de amostras a gerar")
    parser.add_argument("--model_path", type=str, default="generator_client0.pt", help="Caminho do arquivo do gerador salvo")
    args = parser.parse_args()

    images = generate_synthetic(args.class_label, args.num_samples, args.model_path)
    # Exibe
    plt.figure(figsize=(args.num_samples, 1))
    for i, img in enumerate(images):
        plt.subplot(1, args.num_samples, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Label {args.class_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()