"""
ServerApp with agent-controlled federated learning.

Uses FedAvg's evaluate_fn hook to call the agent after each round.
"""

import json
import torch
from flwr.app import ArrayRecord, ConfigRecord, RecordDict
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context

from pytorchexample.task import Net, test as test_fn

app = ServerApp()

# Global state for agent
agent_history = []
round_log = []


def get_evaluate_fn():
    """Return a server-side evaluation function that also calls the agent."""

    def evaluate(round_num, arrays):
        """Evaluate global model and call the agent."""
        # Load model with current global weights
        model = Net()
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Evaluate on a test set
        from torch.utils.data import DataLoader
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import IidPartitioner
        from torchvision.transforms import Compose, ToTensor, Normalize

        # Load full test set
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": IidPartitioner(num_partitions=10)},
        )
        test_set = fds.load_split("test")
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        test_set = test_set.with_transform(apply_transforms)
        testloader = DataLoader(test_set, batch_size=32)

        loss, accuracy = test_fn(model, testloader, device)

        print(f"\n  [Server Eval] Round {round_num}: accuracy={accuracy:.4f}, loss={loss:.4f}")

        # --- AGENT DECISION ---
        global agent_history, round_log
        try:
            from agent import call_agent

            global_metrics = {"eval_acc": round(accuracy, 4), "eval_loss": round(loss, 4)}

            # We don't have per-client metrics in evaluate_fn, so we create a summary
            client_metrics = [{"client_id": i, "train_loss": None, "num_examples": 0}
                              for i in range(10)]

            decision = call_agent(
                round_num=round_num,
                total_rounds=10,
                client_metrics=client_metrics,
                global_metrics=global_metrics,
                history=agent_history,
            )

            agent_history.append({
                "round": round_num,
                "eval_acc": global_metrics["eval_acc"],
                "eval_loss": global_metrics["eval_loss"],
            })

            round_log.append({
                "round": round_num,
                "global_metrics": global_metrics,
                "decision": decision,
            })

            print(f"  [Agent] Select clients {decision['selected_clients']} for next round")
            print(f"  [Agent] Stop early: {decision['stop_early']}")
            print(f"  [Agent] Reasoning: {decision.get('reasoning', 'N/A')}")

            # Save log after each round
            with open("agent_log.json", "w") as f:
                json.dump(round_log, f, indent=2)

        except Exception as e:
            print(f"  [Agent] Error calling agent: {e}")

        return {"loss": loss, "accuracy": accuracy}

    return evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with server-side evaluation
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
    )

    print(f"\n[Server] Starting Agent-Orchestrated FL")
    print(f"[Server] Rounds: {num_rounds}, LR: {lr}\n")

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(),
    )

    # Save final model
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Save final log
    with open("agent_log.json", "w") as f:
        json.dump(round_log, f, indent=2)
    print(f"Saved agent_log.json with {len(round_log)} rounds")
    print("Done!")
