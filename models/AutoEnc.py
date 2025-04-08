import torch
import torch.nn as nn

# HERE IS OUR ARCHITECTURE FOR THE SELF-SUPERVISED AUTOENCODER
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2)

        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2),

        )
    def forward(self, _):
        _ = self.encoder(_)
        _ = self.decoder(_)
        return _
