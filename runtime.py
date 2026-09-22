"""Shared reproducibility and device helpers for the release entry points."""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This release requires an NVIDIA CUDA GPU.")
    index = device.index if device.index is not None else 0
    if index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is not available: %s" % index)
    torch.cuda.set_device(index)
    return index
