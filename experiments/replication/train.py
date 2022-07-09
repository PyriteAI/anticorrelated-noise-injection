from pathlib import Path
from typing import Any, Callable, Optional, Union

import pytorch_lightning as pl
import torchmetrics
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from acni import ACNI

import resnet

HERE = Path(__file__).parent.absolute()


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset = None
        self.valset = None

    def prepare_data(self) -> None:
        CIFAR10(root=HERE / "data", train=True, download=True)
        CIFAR10(root=HERE / "data", train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = CIFAR10(
            root=HERE / "data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        self.valset = CIFAR10(
            root=HERE / "data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers
        )


class ACNIModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = resnet.ResNet18()
        self.val_acc = torchmetrics.Accuracy(num_classes=10)
        self._acni = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.05, momentum=0.9)

        self._acni = ACNI(self.parameters(), std=0.01)

        return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,  # : Union[optim.Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
        )
        self._acni.step()


def main():
    dm = CIFAR10DataModule(batch_size=16, num_workers=4)
    model = ACNIModel()
    trainer = pl.Trainer(devices=1, accelerator="auto", precision=16, max_epochs=500)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
