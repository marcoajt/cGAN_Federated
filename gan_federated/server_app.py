# server_app.py
"""gan-federated: A Flower / PyTorch cGAN server app com agregação de losses."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from gan_federated.task import Generator, Discriminator, get_weights

def weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [n * m.get("accuracy", 0.0) for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"accuracy": sum(accuracies) / total}

def aggregate_fit_losses(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    avg_G = sum(n * m.get("loss_G", 0.0) for n, m in metrics) / total
    avg_D = sum(n * m.get("loss_D", 0.0) for n, m in metrics) / total
    return {"loss_G": avg_G, "loss_D": avg_D}

def server_fn(context: Context) -> ServerAppComponents:
    # Lê configurações
    rounds = context.run_config["num-server-rounds"]
    frac_fit = context.run_config["fraction-fit"]

    # Inicializa pesos globais
    latent_dim, n_classes, img_shape = 100, 10, (1, 28, 28)
    gen = Generator(latent_dim, n_classes, img_shape)
    disc = Discriminator(n_classes, img_shape)
    ndarrays = get_weights(gen) + get_weights(disc)
    params = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=frac_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        fit_metrics_aggregation_fn=aggregate_fit_losses,
    )
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Cria e expõe o ServerApp
app = ServerApp(server_fn=server_fn)
