import torch
import torch.nn.functional as F
from torch import nn
from lightning import LightningModule

class EEGClassifier(LightningModule):
    def __init__(self, num_classes):
        super(EEGClassifier, self).__init__()

        # Define a simple CNN architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.5)

        # Adjust the fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 50, 256)  # Adjust input size to 44800
        self.fc2 = nn.Linear(256, num_classes)   # num_classes should be 80

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension, so x is [batch_size, 1, 62, 400]
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch_size, 32, 31, 200]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch_size, 64, 15, 100]
        x = self.pool(F.relu(self.conv3(x)))  # Shape: [batch_size, 128, 7, 50]

        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 44800]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer