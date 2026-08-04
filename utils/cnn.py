import torch
import torch.nn as nn
import torch.nn.functional as F
 
class CNN(nn.Module):
    """two strided 3x3 convs (4 and 8 channels respectively) + FC head -> 10 logits."""
 
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
 
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
 
        def conv_out_size(size):
            return (size + 2 * 1 - 3) // 2 + 1
 
        size_after_conv1 = conv_out_size(input_size)
        size_after_conv2 = conv_out_size(size_after_conv1)
        flat_dim = 8 * size_after_conv2 * size_after_conv2
 
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits