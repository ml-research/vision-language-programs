import torch


def reserve_gpus():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA devices")

    tensors = []
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        print(f"Allocating tensor on {device}")
        t = torch.zeros((1,), device=device)