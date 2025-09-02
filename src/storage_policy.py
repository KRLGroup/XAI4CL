from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer
)

import torch
import torch.nn.functional as F
from typing import (
    Any,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from gradcam import GradCAM

class ExpandedReservoirSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int, type: str = 'std', patch_size: int = 28, stride: int = 1, epr_selection: bool = True):
        """
        :param max_size: maximum number of samples to be stored
        :param type    : type of replay buffer (std, rrr, epr)
        """
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)
        self._type = type

        if self._type == 'rrr':
            self.saliency_buffer = None
        elif self._type == 'epr':
            self.patch_buffer = None
            self.patch_size = patch_size
            self.stride = stride
            self.top_left_coords = None
            self.epr_selection = epr_selection
    
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        self.post_adapt(strategy, strategy.experience)

    def post_adapt(self, exp, strategy):
        """Update buffer."""
        self.update_from_dataset(strategy, exp.dataset)

    def compute_saliencies(self, strategy: "SupervisedTemplate", dataset: AvalancheDataset):
        """Compute the saliency of the given dataset."""
        explainer = GradCAM(strategy.model, strategy.device, upsample=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=strategy.train_mb_size, shuffle=False)

        saliencies = []
        strategy.model.eval()
        for batch_x, batch_y, _ in data_loader:
            batch_x = batch_x.to(device=strategy.device)
            logits = strategy.model(batch_x)
            pred_probs = torch.softmax(logits, dim=-1).detach().cpu()
            true_prob = torch.gather(pred_probs, 1, torch.unsqueeze(batch_y, dim=-1))
            with torch.set_grad_enabled(True):
                saliency, _, _, _ = explainer(batch_x, strategy.model)
            saliencies.append(saliency.detach().cpu())
        strategy.model.train()
        explainer.remove_hook()
        return torch.cat(saliencies, 0)
    
    def compute_patches(self, strategy: "SupervisedTemplate", dataset: AvalancheDataset, saliencies: torch.Tensor):
        """Compute the most activated patches of the given saliencies."""
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=strategy.train_mb_size, shuffle=False)
        patches, top_left_coords = [], []
        for i, (batch_image, _, _) in enumerate(data_loader):
            # Average pool the saliencies
            saliency = saliencies[i*strategy.train_mb_size:(i+1)*strategy.train_mb_size]
            saliency = torch.unsqueeze(saliency, dim=1)

            # Upsample the saliency to the original image size
            saliency = F.interpolate(
                saliency, batch_image.shape[-2:], mode="bilinear", align_corners=False
            )
            B, C, H, W = saliency.shape
            saliency = saliency.view(B, -1)
            saliency -= saliency.min(dim=1, keepdim=True)[0]
            saliency /= saliency.max(dim=1, keepdim=True)[0]
            saliency = saliency.view(B, C, H, W)

            # Average pool the saliency
            pooled_saliency = F.avg_pool2d(saliency, kernel_size=self.patch_size, stride=self.stride)

            # Find max value and its index in each saliency map
            max_values, indices = torch.max(pooled_saliency.view(pooled_saliency.size(0), -1), dim=1)

            # Convert indices to 2D coordinates
            max_indices_2d = torch.unravel_index(indices, pooled_saliency.shape[-2:])
            top_left_x = (max_indices_2d[-2] - 1) * self.stride
            top_left_y = (max_indices_2d[-1] - 1) * self.stride
            # Ensure valid coordinates (no -1 values)
            top_left_x = torch.clamp(top_left_x, 0, H - self.patch_size)
            top_left_y = torch.clamp(top_left_y, 0, W - self.patch_size)

            for j, (x, y) in enumerate(zip(top_left_x, top_left_y)):
                x_coord = x.item()
                y_coord = y.item()
                top_left_coords.append((x_coord, y_coord))
                patches.append(batch_image[j, :, x_coord:x_coord+self.patch_size, y_coord:y_coord+self.patch_size])
        
        return patches, top_left_coords

    def update_from_dataset(self, strategy: "SupervisedTemplate", new_data: AvalancheDataset):
        """Update the buffer using the given dataset.
        :param new_data:
        :return:
        """

        new_weights = torch.rand(len(new_data))
        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = new_data.concat(self.buffer)

        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        buffer_idxs = sorted_idxs[: self.max_size]

        self.buffer = cat_data.subset(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

        if self._type == 'rrr':
            new_saliencies = self.compute_saliencies(strategy, new_data)
            if not self.saliency_buffer:
                self.saliency_buffer = new_saliencies
                cat_saliencies = new_saliencies
            else:
                cat_saliencies = torch.cat([new_saliencies, self.saliency_buffer])
            self.saliency_buffer = torch.index_select(cat_saliencies, 0, torch.LongTensor(buffer_idxs))
        elif self._type == 'epr':
            new_saliencies = self.compute_saliencies(strategy, new_data)
            new_patches, new_coords = self.compute_patches(strategy, new_data, new_saliencies)
            if not self.patch_buffer:
                self.patch_buffer = new_patches
                cat_patches = new_patches
                self.top_left_coords = new_coords
                cat_coords = new_coords
            else:
                cat_patches = new_patches + self.patch_buffer
                cat_coords = new_coords + self.top_left_coords
            self.patch_buffer = [cat_patches[i] for i in buffer_idxs]
            self.top_left_coords = [cat_coords[i] for i in buffer_idxs]

    def select_patches(self, strategy: Any, img_size: int, num_channels: int = None):
        """ Select patches to store based on prediction correctness, based on EPR paper
            (https://openaccess.thecvf.com/content/WACV2023/html/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.html)
        """
        padded_patches = []
        for i, (_, y, _) in enumerate(self.buffer):
            patch = self.patch_buffer[i]
            top_left_coord = self.top_left_coords[i]
            padded_patch = torch.zeros((num_channels, img_size, img_size), dtype=patch.dtype)
            padded_patch[:, top_left_coord[0]:top_left_coord[0]+patch.shape[1], top_left_coord[1]:top_left_coord[1]+patch.shape[2]] = patch
            padded_patches.append(padded_patch)
        padded_patches = torch.stack(padded_patches, dim=0)
    
        strategy.model.eval()
        out = strategy.model(padded_patches.to(strategy.device))
        strategy.model.train()
        pred_probs = torch.softmax(out, dim=-1).cpu()
        top3_preds = torch.topk(pred_probs, k=3, dim=-1).indices

        pred_flags = torch.tensor(2*len(self.buffer), dtype=torch.int32)
        pred_flags = torch.where(top3_preds == y, 0, pred_flags) # Correct prediction
        pred_flags = torch.where(torch.isin(y, top3_preds), 1, pred_flags).numpy().tolist()  # Correct class among top 3 predictions
            
        # Order elements in patch_buffer and top_left_coords based on pred_flags
        sorted_indices = sorted(range(len(pred_flags)), key=lambda i: pred_flags[i])
        self.patch_buffer = [self.patch_buffer[i] for i in sorted_indices][: self.max_size]
        self.top_left_coords = [self.top_left_coords[i] for i in sorted_indices][: self.max_size]

    def resize(self, strategy: Any, new_size: int, img_size: int = None, num_channels: int = None):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = self.buffer.subset(torch.arange(self.max_size))
        if self._type == 'rrr':
            self.saliency_buffer = self.saliency_buffer[: self.max_size]
        elif self._type == 'epr':
            if self.epr_selection:
                self.select_patches(strategy, img_size, num_channels)
            else:
                self.patch_buffer = self.patch_buffer[: self.max_size]
                self.top_left_coords = self.top_left_coords[: self.max_size]

        self._buffer_weights = self._buffer_weights[: self.max_size]


class ExpandedExperienceBalancedBuffer(ExperienceBalancedBuffer):
    """Rehearsal buffer with samples+saliencies balanced over experiences.
    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self,
                 max_size: int,
                 adaptive_size: bool = True,
                 num_experiences=None,
                 patch_size: int = 28,
                 stride: int = 1,
                 benchmark: str = None,
                 n_tasks: int = 5,
                 type: str = 'std',
                 epr_selection: bool = True,
                 seed: int = 0):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

        self.patch_size = patch_size
        self.stride = stride
        self.benchmark = benchmark
        self.n_tasks = n_tasks
        self.type = type
        self.epr_selection = epr_selection
        self.seed = seed

    @property
    def saliency_buffer(self):
        return torch.cat([g.saliency_buffer for g in self.buffer_groups.values()], 0)
    
    @property
    def patch_buffer(self):
        patches = []
        for g in self.buffer_groups.values():
            patches += g.patch_buffer
        return patches
    
    @property
    def top_left_coords_buffer(self):
        top_left_coords = []
        for g in self.buffer_groups.values():
            top_left_coords += g.top_left_coords
        return top_left_coords

    def post_adapt(self, strategy: "SupervisedTemplate", exp):
        assert strategy.experience is not None
        new_data = exp.dataset
        num_channels, img_size, _ = new_data[0][0].shape[-3:]
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ExpandedReservoirSamplingBuffer(lens[-1], type=self.type, epr_selection=self.epr_selection)
        new_buffer.update_from_dataset(strategy, new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll, img_size=img_size, num_channels=num_channels)
