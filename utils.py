import torch 

def get_device():
    """
    Get the device to be used for training and inference.
    Returns:
        device (torch.device): The device to be used (CPU or GPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
