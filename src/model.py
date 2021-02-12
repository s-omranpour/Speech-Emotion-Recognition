import torch
from torch import nn
from torch.optim import Adam

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

class EmotionRecognizer(LightningModule):
    def __init__(self, in_feat=124, num_classes=16, p_dropout=0.1, lr=1e-4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=p_dropout)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*(in_feat//8 + 1), num_classes)
        )
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
    
    def forward(self, x):
        return self.head(self.conv(x))

    def step(self, batch, mode='train'):
        x, y = batch
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        acc = self.accuracy(pred, y)
        self.log(mode+'_loss', loss.item(), on_step=True)
        self.log(mode+'_acc', acc.item(), on_step=True)
        return {'loss': loss, 'acc':acc}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        return opt


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

