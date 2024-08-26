import matplotlib.pyplot as plt

def plot_final(exp_metrics, n_tasks, checkpoint_dir):
    # Get mean accuracies at the end of each experience
    exp_mean_accs = [sum(exp_metrics['Acc'][exp])/len(exp_metrics['Acc'][exp]) for exp in range(n_tasks)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(exp_mean_accs)), exp_mean_accs, marker='o')
    plt.title(f'Mean accuracy over seen tasks')
    plt.xlabel('Tasks')
    plt.ylabel('Accuracy')
    plt.savefig(checkpoint_dir+'/final_plot_mean_accuracy.png')
    plt.close()