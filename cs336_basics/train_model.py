import typing

import torch
import os

import numpy.typing as npt
import numpy as np

from torch import Tensor


def data_loader(
    dataset: npt.NDArray, 
    batch_size:int, 
    context_length:int, 
    device: str
) -> tuple[Tensor, Tensor]:

    # Sample integers for the starting index of the input sequence
    input_start_indices = np.random.randint(
        low=0, high=(dataset.size - context_length), size=batch_size
    )
    
    inputs = []
    targets = []

    for idx in input_start_indices:
        inputs.append(dataset[idx : idx + context_length])
        targets.append(dataset[idx + 1 : idx + context_length + 1])

    input_batch = np.stack(inputs)
    target_batch = np.stack(targets)

    input_tensor = torch.from_numpy(input_batch).to(dtype=torch.long, device=device)
    target_tensor = torch.from_numpy(target_batch).to(dtype=torch.long, device=device)

    return input_tensor, target_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()

    obj = {
        "model_state_dict": model_state_dict,
        "optim_state_dict": optim_state_dict,
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.nn.Module
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optim_state_dict"])
    iteration = obj["iteration"]
    return iteration


if __name__ == "__main__":
    x = np.array([4, 7, 3, 11, 37, 23])
    inputs, targets = data_loader(dataset=x, batch_size=1, context_length=5, device="mps")
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
