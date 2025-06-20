import argparse

import pytorch_lightning as pl
import torch
import torchvision.models as tvm
from data import ImageNetMultiCropDataModule
from dino_module import DINOHead, DINOLightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# --------------------------- helper to wrap a backbone ------------------------ #
def resnet50_dino(out_dim=65_536, pretrained=True):
    resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
    backbone.out_dim = 2048
    class ResNetWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.projector = DINOHead(in_dim=backbone.out_dim, out_dim=out_dim)
        def forward(self, x):
            feat = self.backbone(x).flatten(1)
            return self.projector(feat)
    return ResNetWrapper()

# ----------------------------------- CLI ------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="/path/to/imagenet")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()

    # student & teacher --------------------------------------------------------
    student = resnet50_dino()
    teacher = resnet50_dino()

    # Lightning objects --------------------------------------------------------
    module = DINOLightningModule(
        student,
        teacher,
        total_epochs=args.epochs,
    )
    datamodule = ImageNetMultiCropDataModule(args.data, batch_size=args.batch)

    # Callbacks ----------------------------------------------------------------
    ckpt = ModelCheckpoint(save_last=True, every_n_epochs=10)
    lrmon = LearningRateMonitor(logging_interval="step")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"                                                  # elsewhere for warm-up etc.

    # 2️⃣ hand that to the Trainer
    trainer = pl.Trainer(
        accelerator=device,         # "cpu" | "mps" | "gpu"
        devices=args.gpus,
        strategy="auto",
        enable_progress_bar=True,       # still shows pbar
        precision="32-true",       # "32-true", "bf16-mixed", …
        callbacks=[ckpt, lrmon],
        log_every_n_steps=50,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    print("RUN")
    main()
