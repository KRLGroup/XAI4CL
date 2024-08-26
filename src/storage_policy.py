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

class ReservoirSamplingBuffer(ExemplarsBuffer):
    """
    Buffer updated with reservoir sampling. Adapted from
    https://github.com/ContinualAI/avalanche/blob/d752103e838babda2e22e2771d5630561c9496c0/avalanche/training/storage_policy.py
    """

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
        """Update buffer."""
        self.post_adapt(strategy, strategy.experience)

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
            if i == 0:
                print(f'NUM CHANNELS: {C}')
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

            for j, (x, y) in enumerate(zip(top_left_x, top_left_y)):
                x_coord = x.item()
                y_coord = y.item()
                top_left_coords.append((x_coord, y_coord))
                patches.append(batch_image[j, :, x_coord:x_coord+self.patch_size, y_coord:y_coord+self.patch_size])
        
        return patches, top_left_coords

    def post_adapt(self, agent, exp):
        """Update buffer."""
        self.update_from_dataset(exp.dataset)

    def update_from_dataset(self,
                            new_data: AvalancheDataset,
                            new_saliencies: torch.Tensor = None,
                            new_patches: Sequence[Any] = None,
                            new_coords: Sequence[Any] = None):
        
        """Update the buffers using the given dataset."""
        
        new_weights = torch.rand(len(new_data))
        cat_weights = torch.cat([new_weights, self._buffer_weights])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        buffer_idxs = sorted_idxs[: self.max_size]

        cat_data = new_data.concat(self.buffer)
        self.buffer = cat_data.subset(buffer_idxs)

        if self._type == 'rrr':
            if not self.saliency_buffer:
                self.saliency_buffer = new_saliencies
                cat_saliencies = new_saliencies
            else:
                cat_saliencies = torch.cat([new_saliencies, self.saliency_buffer])
            self.saliency_buffer = torch.index_select(cat_saliencies, 0, torch.LongTensor(buffer_idxs))

        elif self._type == 'epr':
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

        self._buffer_weights = sorted_weights[: self.max_size]
    
    def select_patches(self, strategy: Any, img_size: int, num_channels: int = None):
        ''' Select patches to store based on prediction correctness, based on EPR paper
            (https://openaccess.thecvf.com/content/WACV2023/html/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.html)
        '''
        pred_flags = []
        for i, (_, y, _) in enumerate(self.buffer):
            patch = self.patch_buffer[i]
            top_left_coord = self.top_left_coords[i]
            padded_patch = torch.zeros((num_channels, img_size, img_size), dtype=patch.dtype)
            padded_patch[:, top_left_coord[0]:top_left_coord[0]+patch.shape[1], top_left_coord[1]:top_left_coord[1]+patch.shape[2]] = patch
            out = strategy.model(torch.unsqueeze(padded_patch, dim=0).to(strategy.device))
            pred_probs = torch.softmax(out[0], dim=-1).cpu()
            top3_preds = torch.topk(pred_probs, k=3).indices.numpy().tolist()

            if top3_preds[0] == y:
                pred_flags.append(0) # Correct prediction
            elif y in top3_preds:
                pred_flags.append(1)  # Correct class among top 3 predictions
            else:
                pred_flags.append(2)
            
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


class RRRExperienceBalancedBuffer(ExperienceBalancedBuffer):
    """Rehearsal buffer with samples and the corresponding saliencies 
    (see https://github.com/SaynaEbrahimi/Remembering-for-the-Right-Reasons).
    
    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True, num_experiences=None):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

        self._num_exps = 0

    @property
    def saliency_buffer(self):
        return torch.cat([g.saliency_buffer for g in self.buffer_groups.values()], 0)

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        new_data = exp.dataset
        lens = self.get_group_lengths(self._num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1], type='rrr')
        new_saliencies = new_buffer.compute_saliencies(agent, new_data)
        new_buffer.update_from_dataset(new_data, new_saliencies=new_saliencies)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(agent, ll)

class EPRExperienceBalancedBuffer(ExperienceBalancedBuffer):
    """Rehearsal buffer with most relevant patches 
    (see https://openaccess.thecvf.com/content/WACV2023/html/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.html).
    
    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True, num_experiences=None, patch_size: int = 28, stride: int = 1,):
        """
        :param max_size: max number of total samples in the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size=max_size, adaptive_size=adaptive_size, num_experiences=num_experiences)

        self._num_exps = 0

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

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        new_data = exp.dataset
        for i, (image, y, t) in enumerate(new_data):
            c, img_size, _ = image.shape[-3:]
            break
        lens = self.get_group_lengths(self._num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1], type='epr')
        new_saliencies = new_buffer.compute_saliencies(agent, new_data)
        new_patches, new_coords = new_buffer.compute_patches(agent, new_data, new_saliencies)
        new_buffer.update_from_dataset(new_data, new_patches=new_patches, new_coords=new_coords)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(agent, ll, img_size=img_size, num_channels=c)
