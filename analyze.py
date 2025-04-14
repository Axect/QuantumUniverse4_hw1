import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
)


def test_model(model, dl_val, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            total_loss += loss.item()
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return total_loss / len(dl_val), all_preds, all_targets


def main():
    # Test run
    console.print("[bold green]Analyzing the model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    # Load the best model
    # study_name = "Optimize_Template"
    # model, config = load_best_model(project, study_name)
    # device = select_device()
    # model = model.to(device)

    _, _, (total_ds, x, y_true) = load_data()  # Assuming this is implemented in util.py
    dl_total = DataLoader(total_ds, batch_size=10000, shuffle=False)

    total_loss, preds, targets = test_model(model, dl_total, device)
    print(f"Total Loss: {total_loss}")
    y_pred = np.array(preds)
    y_data = np.array(targets).reshape(-1)

    # Plotting
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.scatter(x, y_data, label="Data", alpha=0.1, s=10, color='blue')
        ax.plot(x, y_true, label="True", color='green', linewidth=2)
        ax.plot(x, y_pred, label="Predicted", color='red', linewidth=2, linestyle="--")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.legend()
        ax.autoscale(tight=True)
        fig.savefig("best_result.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    console = Console()
    main()
