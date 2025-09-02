import os
import json
import matplotlib.pyplot as plt

def load_metrics_from_folder(metric_files):
    all_metrics = {}
    for file_path in metric_files:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
            all_metrics[file_path] = metrics
    return all_metrics

def compute_mean_accuracies(metrics_list):
    mean_accs = []
    for i in range(len(metrics_list)):
        accs = []
        for j in range(i + 1):  # Seen tasks so far
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}"
            acc = metrics_list[i].get(key, None)
            if acc is not None:
                accs.append(acc)
        mean_accs.append(sum(accs) / len(accs) if accs else 0.0)
    return mean_accs

def final_accuracy(metrics_list):
    last = metrics_list[-1]
    accs = [v for k, v in last.items() if k.startswith("Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp")]
    return sum(accs) / len(accs) if accs else 0.0

def plot_results(all_metrics, filename_to_method):
    plt.figure(figsize=(10, 6))
    final_accs = {}

    for filename, metrics in all_metrics.items():
        mean_accuracies = compute_mean_accuracies(metrics)
        plt.plot(range(1, len(mean_accuracies)+1, 1), mean_accuracies, label=filename_to_method[filename])
        final_accs[filename] = final_accuracy(metrics)

    plt.xlabel("Trained Task Index", fontsize=18)
    plt.ylabel("Mean accuract over seen tasks", fontsize=18)
    plt.title("Split-CIFAR-100", fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(range(1,11), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mean_accuracy_plot.pdf", dpi=1200)

    print("\nFinal Mean Accuracies on Test Stream:")
    for filename, acc in final_accs.items():
        print(f"{filename}: {acc:.4f}")

# Provide the path to your folder containing the JSON files
filename_to_method = {
    "./checkpoints/cifar100-split_naive_1824/test_results.json": "Naive Finetuning",
    "./checkpoints/cifar100-split_rrr_1824/test_results.json": "RRR"
}
all_metrics = load_metrics_from_folder(filename_to_method.keys())
plot_results(all_metrics, filename_to_method)
