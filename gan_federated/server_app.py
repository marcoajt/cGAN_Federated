"""gan-federated: Flower / PyTorch cGAN server app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from gan_federated.task import (
    Generator,
    Discriminator,
    get_parameters,
    weighted_average_loss,
)

def server_fn(context: Context):
    # Lê do config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Parâmetros iniciais: G + D
    gen = Generator()
    disc = Discriminator()
    init_ndarrays = get_parameters(gen) + get_parameters(disc)
    init_params = ndarrays_to_parameters(init_ndarrays)

    # Estratégia FedAvg
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=init_params,
        evaluate_metrics_aggregation_fn=weighted_average_loss,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Inicia o server app
app = ServerApp(server_fn=server_fn)
