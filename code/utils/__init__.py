from torch import Tensor
import torch
import torch.cuda

def use_same_device_as_input_tensor(input_tensor: Tensor=None) -> str:
    if input_tensor is None: return get_cuda_device_if_available()

    return input_tensor.get_device()

def get_cuda_device_if_available() -> str:
    if torch.cuda.is_available(): 'cuda'
    return 'cpu'
