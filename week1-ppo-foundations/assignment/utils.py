import os
import json

import matplotlib.pyplot as plt


def _get_run_prefix(run_name):
    return run_name if run_name else "default_run"


def plot_learning_curve(update_idx, eval_avg_history, eval_runs_per_update, run_name=""):
    plt.figure(figsize=(8, 4.5))
    plt.plot(update_idx, eval_avg_history, marker="o", linewidth=1.5)
    plt.xlabel("Update Loop")
    plt.ylabel(f"Average Reward ({eval_runs_per_update} eval runs)")
    plt.title("PPO Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"{_get_run_prefix(run_name)}_learning_curve.png",
    )
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curve to: {out_path}")


def save_run_metrics(run_name, update_idx, eval_avg_history, eval_rewards_history):
    payload = {
        "run_name": _get_run_prefix(run_name),
        "update_idx": update_idx,
        "eval_avg_history": eval_avg_history,
        "eval_rewards_history": eval_rewards_history,
    }
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"{_get_run_prefix(run_name)}_eval_metrics.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved eval metrics to: {out_path}")
