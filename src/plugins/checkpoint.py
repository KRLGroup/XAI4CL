import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from avalanche.training.plugins import SupervisedPlugin


class MetricsCheckpoint(SupervisedPlugin):
    """
    Create dictionaries with loss and accuracy history.
    """

    def __init__(
        self,
        n_tasks,
        checkpoint_dir,
        store_plots=True,
        checkpoint_every=1,
        verbose=False,
    ):
        """Init:
        :param store_plots:             if True, store plots of metrics history
        :param checkpoint_every:        when > 0, store metrics plots every checkpoint_every epochs
                                        otherwise store plots at the end of each experience. If it
                                        is -1, store plots at the end of each task.
        :param checkpoint_dir:          directory where to save metrics plots
        """
        super().__init__()

        self.n_tasks = n_tasks
        self.store_plots = store_plots
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose

        self.checkpoint_dir = checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'Checkpoint directory: {self.checkpoint_dir}')

        self.metrics_history = {'Loss': {}, 'Acc': {}, 'AvgAcc': []}
        self.exp_metrics = {'Loss': {}, 'Acc': {}, 'BWT': {}, 'FWT': {}}
        self.check_training_started = 0

    def before_training_exp(self, strategy, **kwargs):
        self.check_training_started = 1

    def after_training_epoch(self, strategy, **kwargs):
        current_epoch = strategy.clock.train_exp_epochs
        self._update_metrics_history(strategy)
        if current_epoch % self.checkpoint_every == 0:
            self._store_metrics_plots(strategy.clock.train_exp_counter)
            if self.verbose:
                print(f'Experience {strategy.clock.train_exp_counter}, epoch {strategy.clock.train_exp_epochs}: Metrics plots saved! ')
    
    
    def after_training_exp(self, strategy, **kwargs):
        self._update_exp_metrics(strategy)
        if self.verbose:
            print(f'End of experience {strategy.clock.train_exp_counter}')


    def _update_exp_metrics(self, strategy):
        """Stores test losses, accuracies, BWTs, and FWTs at the end of each experience.

        Args:
            strategy: Strategy object.
        """

        # Get current experience being trained
        if self.check_training_started:
            current_exp = strategy.clock.train_exp_counter - 1
        else:
            current_exp = strategy.clock.train_exp_counter

        # Get all metrics
        eval_metrics = strategy.evaluator.get_last_metrics()

        for exp in range(0, self.n_tasks):
            # Update loss and accuracy history for individual experiences
            if current_exp not in self.exp_metrics['Loss'].keys():
                self.exp_metrics['Loss'][current_exp] = []
                self.exp_metrics['Acc'][current_exp] = []
                self.exp_metrics['BWT'][current_exp] = []
                self.exp_metrics['FWT'][current_exp] = []


            base_metric_name = '_Exp/eval_phase/valid_stream/Exp00' + str(exp)
            acc_metric_name = 'Top1_Acc' + base_metric_name
            loss_metric_name = 'Loss' + base_metric_name
            self.exp_metrics['Acc'][current_exp].append(round(eval_metrics[acc_metric_name], 2))
            self.exp_metrics['Loss'][current_exp].append(round(eval_metrics[loss_metric_name], 2))

            if exp > current_exp:
                fwt_name = 'ExperienceForwardTransfer/eval_phase/valid_stream/Exp00' + str(exp)
                self.exp_metrics['FWT'][current_exp].append(round(eval_metrics[fwt_name], 2))
                self.exp_metrics['BWT'][current_exp].append(0.0)
            elif exp < current_exp:
                bwt_name = 'ExperienceBWT/eval_phase/valid_stream/Exp00' + str(exp)
                self.exp_metrics['FWT'][current_exp].append(0.0)
                self.exp_metrics['BWT'][current_exp].append(round(eval_metrics[bwt_name], 2))
            else:
                self.exp_metrics['FWT'][current_exp].append(0.0)
                self.exp_metrics['BWT'][current_exp].append(0.0)


    def _update_metrics_history(self, strategy):
        """Stores validation loss and accuracy for each experience every checkpoint_every epochs.

        Args:
            strategy: Strategy object.
        """
        # Get all metrics
        eval_metrics = strategy.evaluator.get_last_metrics()
        # Get current experience being trained
        current_exp = strategy.clock.train_exp_counter

     
        mean_acc = 0
        for exp in range(self.n_tasks):
            # Update loss and accuracy history for individual experiences
            if exp not in self.metrics_history['Loss'].keys():
                self.metrics_history['Loss'][exp] = {}
                self.metrics_history['Acc'][exp] = {}
            if current_exp not in self.metrics_history['Loss'][exp].keys():
                self.metrics_history['Loss'][exp][current_exp] = []
                self.metrics_history['Acc'][exp][current_exp] = []

            # Get metrics names
            base_metric_name = '_Exp/eval_phase/valid_stream/Exp00' + str(exp)
            acc_metric_name = 'Top1_Acc' + base_metric_name
            loss_metric_name = 'Loss' + base_metric_name
            
            # Compute accuracy and update history
            accuracy = round(eval_metrics[acc_metric_name], 2)
            self.metrics_history['Acc'][exp][current_exp].append(accuracy)
            
            # Compute mean accuracy over seen tasks
            if exp <= current_exp:
                mean_acc += accuracy
            
            # Update loss history
            self.metrics_history['Loss'][exp][current_exp].append(round(eval_metrics[loss_metric_name], 2))
        
        self.metrics_history['AvgAcc'].append(mean_acc / (current_exp + 1))


    def _store_metrics_plots(self, last_trained_exp):
        if self.store_plots:
            colors = ['steelblue','darkviolet','orangered','yellowgreen','turquoise']

            # Plot loss history
            plt.figure(figsize=(10, 6))
            for i, exp in enumerate(self.metrics_history['Loss'].keys()):
                y = self.metrics_history['Loss'][exp][last_trained_exp]
                x = range(len(y))
                plt.plot(x, y, label=f'Exp {exp}', color=colors[i % len(colors)], marker= 's', alpha=0.7) if len(y)>1 else plt.plot(x, y, 's', label=f'Loss exp {exp}', color=colors[i % len(colors)], alpha=0.7)
            plt.title(f'Loss history on experience {last_trained_exp}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(self.checkpoint_dir + f'/loss_history_exp_{last_trained_exp}.png')
            plt.close()

            # Plot accuracy history
            plt.figure(figsize=(10, 6))
            for i, exp in enumerate(self.metrics_history['Acc'].keys()):
                y = self.metrics_history['Acc'][exp][last_trained_exp]
                x = range(len(y))
                plt.plot(x, y, label=f'Exp {exp}', marker='o', color=colors[i % len(colors)], alpha=0.7) if len(y)>1 else plt.plot(x, y, 'o', label=f'Accuracy exp {exp}', color=colors[i % len(colors)], alpha=0.7)
            plt.title(f'Accuracy history on experience {last_trained_exp}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(self.checkpoint_dir + f'/accuracy_history_exp_{last_trained_exp}.png')
            plt.close()
            
            # Plot mean accuracy over seen tasks history
            plt.figure(figsize=(10, 6))
            accuracy_values = self.metrics_history['AvgAcc']
            plt.plot(range(len(accuracy_values)), accuracy_values, marker='o')
            plt.title(f'Mean accuracy over seen tasks')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.savefig(self.checkpoint_dir + f'/mean_accuracy_history_exp_{last_trained_exp}.png')
            plt.close()