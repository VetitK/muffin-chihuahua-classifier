import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from model import MuffinChihuahuaClassifier
from torchmetrics import Accuracy
import wandb
class MuffinChihuahuaLightningModule(LightningModule):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.accuracy = Accuracy(task='binary', num_classes=2)
        self.model = MuffinChihuahuaClassifier()
        self.loss = nn.NLLLoss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        y_hat = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        y_hat = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        y_hat = torch.argmax(y_pred, dim=1)
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
        return {"test_loss": loss, "test_acc": acc}
    