import matplotlib
matplotlib.use('Agg')
import torch
from typing import (
    Any,
    Optional,
    TypeVar,
)

from packaging.version import parse
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks import CLExperience
from gradcam import GradCAM
from storage_policy import RRRExperienceBalancedBuffer

TExperienceType = TypeVar("TExperienceType", bound=CLExperience)


class RRRPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Implements the Remembering for the right reasons, the replay-based
    method proposed in https://openreview.net/pdf?id=tHgJoMfy6nI.

    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_epoch` callback is implemented in order to compute
    the loss on the saliencies, preventing explanation drift.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    :param xai_loss: loss to prevent saliency drift
    :param xai_optimizer: optimizer to update parameters to prevent 
        saliency drift.
    :param xai_regularizer: xai_loss regularization factor.
    :param mem_size: total number of patterns to be stored in the external
        memory.
    :param mem_adaptive_size: whether to use a memory buffer with an 
        adaptive size.
    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    """

    def __init__(
        self,
        xai_loss: torch.nn.Module,
        xai_optimizer: torch.optim.Optimizer,
        xai_regularizer: float = 100.0,
        mem_size: int = 1000,
        mem_adaptive_size: bool = True,
        num_experiences: int = None,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
    ):
        super().__init__()

        assert mem_adaptive_size or num_experiences is not None, \
                "You can either use an adaptive memory (mem_size is divided equally over all observed experiences), \
                or divide mem_size by the fixed number of experiences"

        self.xai_loss = xai_loss
        self.xai_optimizer = xai_optimizer
        self.xai_regularizer = xai_regularizer
        self.mem_size = mem_size
        self.mem_adaptive_size = mem_adaptive_size
        self.num_experiences = num_experiences
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.storage_policy = RRRExperienceBalancedBuffer(self.mem_size, adaptive_size=self.mem_adaptive_size, num_experiences=self.num_experiences)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    def after_training_epoch(self, strategy: Any, *args, **kwargs):
        if strategy.clock.train_exp_counter > 0:
            # Loop over the saliency buffer
            data_loader = torch.utils.data.DataLoader(self.storage_policy.buffer, batch_size=strategy.train_mb_size, shuffle=False)
            explainer = GradCAM(strategy.model, strategy.device, upsample=True)
            for i, (batch_image, _, _) in enumerate(data_loader):
                batch_image = batch_image.to(device=strategy.device)

                # Compute new saliency maps
                with torch.set_grad_enabled(True):
                    new_saliencies, _, _, _ = explainer(batch_image, strategy.model)

                # Compute saliency loss
                old_saliencies = self.storage_policy.saliency_buffer[i*strategy.train_mb_size:(i+1)*strategy.train_mb_size]
                xai_loss = self.xai_loss(old_saliencies.to(strategy.device), new_saliencies)
                xai_loss *= self.xai_regularizer

                self.xai_optimizer.zero_grad()
                xai_loss.backward(retain_graph=True)
                self.xai_optimizer.step()

            explainer.remove_hook()
            del new_saliencies, old_saliencies, xai_loss
            torch.cuda.empty_cache()
    
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """ Update the buffer. """
        self.storage_policy.post_adapt(strategy, strategy.experience)
