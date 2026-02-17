import os

import matplotlib.pyplot as plt


def plot_learning_curve(update_idx, eval_avg_history, eval_runs_per_update):
    plt.figure(figsize=(8, 4.5))
    plt.plot(update_idx, eval_avg_history, marker="o", linewidth=1.5)
    plt.xlabel("Update Loop")
    plt.ylabel(f"Average Reward ({eval_runs_per_update} eval runs)")
    plt.title("PPO Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "learning_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curve to: {out_path}")
