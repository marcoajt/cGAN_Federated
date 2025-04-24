"""gan-federated: A Flower / PyTorch cGAN server app with per-round loss aggregation."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from gan_federated.task import Generator, Discriminator, get_weights


def weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics for evaluate."""
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def aggregate_fit_losses(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate generator and discriminator losses from fit metrics."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_G = sum(num_examples * m.get("loss_G", 0.0) for num_examples, m in metrics) / total_examples
    avg_D = sum(num_examples * m.get("loss_D", 0.0) for num_examples, m in metrics) / total_examples
    return {"loss_G": avg_G, "loss_D": avg_D}


def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    latent_dim = 100
    n_classes = 10
    img_shape = (1, 28, 28)
    gen = Generator(latent_dim, n_classes, img_shape)
    disc = Discriminator(n_classes, img_shape)
    ndarrays = get_weights(gen) + get_weights(disc)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy with fit and evaluate aggregators
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        fit_metrics_aggregation_fn=aggregate_fit_losses,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)