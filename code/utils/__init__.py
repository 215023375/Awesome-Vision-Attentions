from torch import Tensor


def use_same_device_as_input_tensor(input_tensor: Tensor) -> str:
    return input_tensor.get_device()
