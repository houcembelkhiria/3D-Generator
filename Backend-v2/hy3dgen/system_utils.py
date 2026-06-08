import torch

def get_device() -> str:
    """
    Returns the best available device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def empty_cache():
    """
    Empties the hardware cache for the best available device.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def synchronize():
    """
    Synchronizes the hardware streams for the best available device.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
        torch.mps.synchronize()
