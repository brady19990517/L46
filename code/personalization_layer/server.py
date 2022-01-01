import flwr as fl
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server

# DEFAULT_SERVER_ADDRESS = "localhost:8080"

# strategy = fl.server.strategy.FedAvg(
#     min_fit_clients=10,
#     min_eval_clients=10,
#     min_available_clients=10,
# )

# client_manager = fl.server.SimpleClientManager()
# server = fl.server.Server(client_manager=client_manager, strategy=strategy)

# # fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)

# print(f"Starting gRPC server on {DEFAULT_SERVER_ADDRESS}...")
# grpc_server = start_insecure_grpc_server(
#         client_manager=server.client_manager(),
#         server_address=DEFAULT_SERVER_ADDRESS,
#         max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH,
#     )

# print("Fitting the model...")
# hist = server.fit(num_rounds=3)
# test_loss, test_metrics = server.strategy.evaluate(parameters=server.parameters)
# print(f"Server-side test results after training: test_loss={test_loss:.4f}, "
#         f"test_accuracy={test_metrics['accuracy']:.4f}")

# grpc_server.stop(None)    
fl.server.start_server(config={"num_rounds": 3})