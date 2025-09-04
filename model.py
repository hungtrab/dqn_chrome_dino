import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size of flattened features after conv layers
        # For 84x84 input: 84 -> 20 -> 9 -> 7 after conv layers
        # So 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256), 
            nn.ReLU(), 
            nn.Linear(256, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct dimensions for Conv2D
        if len(x.shape) == 3:  # [C, H, W] -> add batch dimension
            x = x.unsqueeze(0)  # [1, C, H, W]
        elif len(x.shape) == 5:  # Remove extra dimensions
            x = x.squeeze()
        
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x