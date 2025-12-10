# src/utils/general_utils.py
import torch
import numpy as np
import random

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # 可选，可能影响性能
        torch.backends.cudnn.benchmark = False   # 可选

def get_device(preferred_cuda_index=1):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > preferred_cuda_index:
            device = torch.device(f"cuda:{preferred_cuda_index}")
            print(f"Using device: cuda:{preferred_cuda_index}")
        else:
            device = torch.device("cuda:0")
            print(f"Warning: cuda:{preferred_cuda_index} not available. Using cuda:0.")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available. Using CPU.")
    return device