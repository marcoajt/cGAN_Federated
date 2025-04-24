"""gan-federated: A Flower / PyTorch cGAN client app."""

import torch
import json
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from gan_federated.task import (
    Generator,
    Discriminator,
    load_dataloader,
    train_cgan,
    evaluate_cgan,
    get_parameters,
    set_parameters,
)

class FlowerClient(NumPyClient):
    def __init__(self, client_id: int, num_partitions: int):
        self.client_id = client_id
        self.num_partitions = num_partitions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_dim = 100
        self.n_classes = 10
        self.img_shape = (1, 28, 28)
        self.local_epochs = 1

        # Load data partition, models
        self.train_loader = load_dataloader(client_id, num_partitions)
        self.generator = Generator(self.latent_dim, self.n_classes, self.img_shape).to(self.device)
        self.discriminator = Discriminator(self.n_classes, self.img_shape).to(self.device)

        # Optimizers and adversarial loss
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.adversarial_loss = torch.nn.BCELoss()

    def get_parameters(self, config):
        return get_parameters(self.generator) + get_parameters(self.discriminator)

    def set_parameters(self, parameters, config):
        gen_len = len(self.generator.state_dict())
        set_parameters(self.generator, parameters[:gen_len])
        set_parameters(self.discriminator, parameters[gen_len:])

    def fit(self, parameters, config):
        """Local cGAN training, recording losses and saving history."""
        self.set_parameters(parameters, config)
        # Unpack generator loss, discriminator loss, and full history
        loss_G, loss_D, history = train_cgan(
            generator=self.generator,
            discriminator=self.discriminator,
            train_loader=self.train_loader,
            optimizer_G=self.optimizer_G,
            optimizer_D=self.optimizer_D,
            adversarial_loss=self.adversarial_loss,
            epochs=self.local_epochs,
            device=self.device,
        )
        # Client 0 salva modelo e histórico
        if self.client_id == 0:
            torch.save(self.generator.state_dict(), "generator_client0.pt")
            with open("loss_history.json", "w") as f:
                json.dump(history, f)
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"loss_G": loss_G, "loss_D": loss_D},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        metrics = evaluate_cgan(
            generator=self.generator,
            discriminator=self.discriminator,
            train_loader=self.train_loader,
            adversarial_loss=self.adversarial_loss,
            device=self.device,
        )
        return float(metrics["loss"]), len(self.train_loader.dataset), {"loss": metrics["loss"]}


def client_fn(context: Context):
    partition_id = int(context.node_config.get("partition-id", context.node_id))
    num_partitions = int(context.node_config.get("num-partitions", partition_id + 1))
    return FlowerClient(partition_id, num_partitions).to_client()

# Inicia o cliente
app = ClientApp(client_fn)
