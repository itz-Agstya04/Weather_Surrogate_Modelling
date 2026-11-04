import torch
import torch.nn as nn


class ResidualStep(nn.Module):
    def __init__(self, net: nn.Module, step_std: torch.Tensor):
        super().__init__()
        values = torch.as_tensor(step_std, dtype=torch.float32).reshape(1, -1, 1, 1)
        self.register_buffer("s", values)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.s.to(dtype=x.dtype, device=x.device) * self.net(x)


def step_loss(pred: torch.Tensor, target: torch.Tensor, step_std: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(step_std, dtype=pred.dtype, device=pred.device).reshape(1, -1, 1, 1)
    return (((pred - target) / scale) ** 2).mean()
