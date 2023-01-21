import torch
import torch.nn as nn
from torchvision.models import resnet18

class MuffinChihuahuaClassifier(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Sequential(torch.nn.Linear(512, 2),
                                         nn.LogSoftmax(dim=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x)