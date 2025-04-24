"""gan-federated: toda a lógica do cGAN + utilitários Flower."""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparâmetros de dados
BATCH_SIZE = 64

# Transformação padrão
_default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Carrega MNIST completo
_full_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=_default_transform
)

def get_partition(dataset, client_id, num_clients):
    total = len(dataset)
    indices = np.arange(total)
    size = total // num_clients
    start = client_id * size
    end = start + size if client_id != num_clients - 1 else total
    return torch.utils.data.Subset(dataset, indices[start:end])

def load_dataloader(client_id: int, num_clients: int):
    subset = get_partition(_full_dataset, client_id, num_clients)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        input_dim = latent_dim + n_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, n_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        input_dim = n_classes + int(np.prod(img_shape))
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)
        return self.model(torch.cat((img_flat, label_embedding), dim=1))

def train_cgan(generator, discriminator, train_loader,
               optimizer_G, optimizer_D, adversarial_loss,
               epochs, device):
    generator.train()
    discriminator.train()
    total_G, total_D = 0.0, 0.0
    history = {"loss_G": [], "loss_D": []}
    for _ in range(epochs):
        for imgs, labels in train_loader:
            bs = imgs.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)
            real = imgs.to(device)
            lbls = labels.to(device)

            optimizer_D.zero_grad()
            loss_real = adversarial_loss(discriminator(real, lbls), valid)
            z = torch.randn(bs, generator.latent_dim, device=device)
            gen_lbls = torch.randint(0, generator.n_classes, (bs,), device=device)
            gen_imgs = generator(z, gen_lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs.detach(), gen_lbls), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            output = discriminator(gen_imgs, gen_lbls)
            loss_G = adversarial_loss(output, valid)
            loss_G.backward()
            optimizer_G.step()

            total_G += loss_G.item()
            total_D += loss_D.item()
            history["loss_G"].append(loss_G.item())
            history["loss_D"].append(loss_D.item())

    n_batches = len(train_loader) * epochs
    avg_G = total_G / n_batches
    avg_D = total_D / n_batches
    return avg_G, avg_D, history

def evaluate_cgan(generator, discriminator, train_loader, adversarial_loss, device):
    generator.eval()
    discriminator.eval()
    total, count = 0.0, 0
    valid = torch.ones(BATCH_SIZE, 1, device=device)
    fake = torch.zeros(BATCH_SIZE, 1, device=device)
    with torch.no_grad():
        for imgs, labels in train_loader:
            bs = imgs.size(0)
            real = imgs.to(device)
            lbls = labels.to(device)
            loss_real = adversarial_loss(discriminator(real, lbls), valid[:bs])
            z = torch.randn(bs, generator.latent_dim, device=device)
            gen_lbls = torch.randint(0, generator.n_classes, (bs,), device=device)
            gen_imgs = generator(z, gen_lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs, gen_lbls), fake[:bs])
            total += (loss_real + loss_fake).item()
            count += 1
    return {"loss": total / count if count > 0 else float("inf")}

def get_parameters(net: nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters):
    params = zip(net.state_dict().keys(), parameters)
    sd = OrderedDict({k: torch.tensor(v) for k, v in params})
    net.load_state_dict(sd, strict=True)

# Alias get_weights para compatibilidade com server_app.py
get_weights = get_parameters

from flwr.common import Metrics

def weighted_average_loss(metrics: list[tuple[int, Metrics]]):
    losses = [n * m["loss"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"loss": sum(losses) / total}