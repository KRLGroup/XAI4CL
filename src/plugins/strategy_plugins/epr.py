import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from typing import (
    Any,
    Optional,
    TypeVar,
)

from packaging.version import parse
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks import CLExperience
from storage_policy import EPRExperienceBalancedBuffer

TExperienceType = TypeVar("TExperienceType", bound=CLExperience)


class EPRPlugin(SupervisedPlugin, supports_distributed=True):
    """
    EPR plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

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
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(
        self,
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

        if mem_adaptive_size:
            num_experiences = None

        self.mem_size = mem_size
        self.mem_adaptive_size = mem_adaptive_size
        self.num_experiences = num_experiences
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.storage_policy = EPRExperienceBalancedBuffer(max_size=self.mem_size, adaptive_size=self.mem_adaptive_size, num_experiences=self.num_experiences)

    def create_patch_dataset(self, patches, top_left_coords, full_sized_dataset):
        padded_patches, ys, ts = [], [], []
        for i, (image, y, t) in enumerate(full_sized_dataset):
            c, img_size, _ = image.shape[-3:]
            patch = patches[i]
            top_left_coord = top_left_coords[i]
            padded_patch = torch.zeros((c, img_size, img_size), dtype=patch.dtype)
            padded_patch[:, top_left_coord[0]:top_left_coord[0]+patch.shape[1], top_left_coord[1]:top_left_coord[1]+patch.shape[2]] = patch
            
            padded_patches.append(torch.unsqueeze(padded_patch, dim=0))
            ys.append(y)
            ts.append(t)

        padded_patches = torch.cat(padded_patches, dim=0)
        ys = torch.tensor(ys)
        ts = torch.tensor(ts)

        torch_data = TensorDataset(padded_patches, ys)
        patch_dataset = _make_taskaware_classification_dataset(torch_data, task_labels=ts)
        
        return patch_dataset

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
        if len(self.storage_policy.patch_buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        
        patch_dataset = self.create_patch_dataset(self.storage_policy.patch_buffer,
                                                  self.storage_policy.top_left_coords_buffer,
                                                  self.storage_policy.buffer)

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
            patch_dataset,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )
    
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """ Update the buffer. """
        self.storage_policy.post_adapt(strategy, strategy.experience)
