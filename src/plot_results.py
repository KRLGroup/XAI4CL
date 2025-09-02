import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils import *

metric_name_dict = {
    "Top1_Acc_Exp": "Accuracy",
    "ExperienceBWT": "Backward Transfer",
    "ExperienceForwardTransfer": "Forward Transfer",
}

def main():
    # Parse arguments
    args = parse_arguments()

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    set_all_seeds(args.seed)
    # Generate args.num_seeds random seeds
    seeds = generate_random_seeds(args.num_seeds)

    seed_results = []
    for seed in seeds:
        checkpoint_dir = args.setup.checkpoint_dir+'/'+args.experiment.benchmark+'_'+args.experiment.strategy+'_'+str(seed)
        results_filename = os.path.join(checkpoint_dir, 'test_results.json')
        assert os.path.exists(results_filename), f"Results file {results_filename} does not exist. Please run the training loop first."
        with open(results_filename, 'r') as f:
            results = json.load(f)
        
        for train_exp_idx, result in enumerate(results):
            for metric, value in result.items():
                if "test_stream" in metric and any(key in metric for key in metric_name_dict):
                    metric = metric.split("/")
                    seed_results.append({
                        "seed": seed,
                        "trained_task": train_exp_idx,
                        "eval_task": int(metric[-1][-1]),
                        "metric": metric_name_dict[metric[0]],
                        "value": value
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(seed_results)
    # Save DataFrame to CSV
    output_csv_path = os.path.join(checkpoint_dir, 'results_summary.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # Pivot the DataFrame for heatmap plotting
    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        pivot_table = metric_df.pivot_table(
            index="trained_task", columns="eval_task", values="value", aggfunc=np.mean
        )

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": metric},
        )
        plt.title(f"{metric} Heatmap")
        plt.xlabel("Evaluation Task")
        plt.ylabel("Trained Task")
        plt.tight_layout()

        # Save the plot
        heatmap_path = os.path.join(checkpoint_dir, f"{metric}_heatmap.png")
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to {heatmap_path}")
        plt.close()

if __name__ == "__main__":
    main()