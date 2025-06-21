import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import LRSchedulerConfig


# ----------------------------------------------------------------------------- #
#  Small helper: cosine schedule for a scalar (teacher momentum & temperature)  #
# ----------------------------------------------------------------------------- #
def cosine_schedule(start: float, end: float, steps: int, step: int) -> float:
    """Cosine interpolation from `start`→`end` over `steps` steps."""
    progress = min(step, steps) / steps
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


# ----------------------------------------------------------------------------- #
#               Projection head (same as Facebook's original repo)             #
# ----------------------------------------------------------------------------- #
class DINOHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int = 2048,
        bottleneck_dim: int = 256, out_dim: int = 65_536,
        nlayers: int = 3, norm_last_layer: bool = True
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(nlayers - 2):
            layers += [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
        layers += [nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.last_layer(x)


# ----------------------------------------------------------------------------- #
#                          LightningModule with DINO loss                       #
# ----------------------------------------------------------------------------- #
class DINOLightningModule(pl.LightningModule):
    """Lightning implementation of DINO self-distillation.

    • `student`  – your encoder + projector (trainable)
    • `teacher`  – EMA copy of the student (no gradients)
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        *,
        out_dim: int = 65_536,
        total_epochs: int = 300,
        warmup_epochs: int = 10,
        base_lr: float = 1e-3,          # will be scaled by batch_size/256
        weight_decay: float = 0.04,
        weight_decay_end: float = 0.4,
        student_temp: float = 0.1,
        teacher_temp_start: float = 0.04,
        teacher_temp_end: float = 0.04,
        teacher_temp_warmup: int = 30_000,
        momentum_base: float = 0.996,
        momentum_end: float = 1.0,
        center_momentum: float = 0.9,
        **optimizer_kwargs,            # extra AdamW args
    ):
        super().__init__()

        # networks ----------------------------------------------------------------
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False  # teacher is frozen

        # hyper-parameters --------------------------------------------------------
        self.save_hyperparameters(ignore=["student", "teacher"])
        self.hp = self.hparams  # shorthand

        # running centre buffer (Alg. 1) -----------------------------------------
        self.register_buffer("center", torch.zeros(1, out_dim))

    # --------------------------------------------------------------------------- #
    #                               forward alias                                 #
    # --------------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # inference
        return self.student(x)

    # --------------------------------------------------------------------------- #
    #                               training step                                 #
    # --------------------------------------------------------------------------- #
    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Just in case Lightning sets it to train mode again
        self.teacher.eval()
        self.student.projector.last_layer.weight_g.requires_grad = False
        
        # Optional: Update weight decay manually
        wd = cosine_schedule(
            self.hparams.weight_decay,
            self.hparams.weight_decay_end,
            self.total_steps,
            self.global_step,
        )
        for param_group in self.optimizers().param_groups:
            param_group["weight_decay"] = wd

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """`batch` is a list of `n_crops` tensors. The first 2 are global crops
        used for the teacher; the rest are local crops.
        """
        crops: list[torch.Tensor] = batch
        n_global = 2
        student_out = [self.student(c).float() for c in crops]

        # ---------------------- teacher forward pass ----------------------------
        with torch.no_grad():
            global_crops = crops[:n_global]
            teacher_out = [self.teacher(c).float() for c in global_crops]

            # center + sharpen
            t_temp = self.teacher_temperature()
            teacher_logits = [(t - self.center) / t_temp for t in teacher_out]
            teacher_probs = [F.softmax(t, dim=-1) for t in teacher_logits]

        # ----------------------------- student side -----------------------------
        s_temp = self.hp.student_temp
        total_loss, n_terms = 0.0, 0
        for iq, t_prob in enumerate(teacher_probs):
            for v in range(len(crops)):
                if v == iq:
                    continue
                s_prob = F.log_softmax(student_out[v] / s_temp, dim=-1)
                total_loss += torch.sum(-t_prob * s_prob, dim=-1).mean()
                n_terms += 1
        loss = total_loss / n_terms
        self.log("dino_loss", loss, on_step=True, prog_bar=True)

        # --------------------- update centre (Alg. 1) ---------------------------
        with torch.no_grad():
            batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center = (
                self.center * self.hp.center_momentum
                + batch_center * (1 - self.hp.center_momentum)
            )

        # If center is stuck near zero or teacher output is very small → collapse risk
        if self.global_step % 100 == 0:
            self.log("center_mean", self.center.mean())
            self.log("teacher_out_mean", torch.cat(teacher_out).mean())

        # ------------------ entropy monitoring ----------------------------------
        if self.global_step % 5 == 0:
            with torch.no_grad():
                t_entropy = torch.stack([self.compute_entropy(t) for t in teacher_out]).mean()
                s_entropy = torch.stack([self.compute_entropy(s) for s in student_out]).mean()
                self.log("teacher_temp", self.teacher_temperature())
                self.log("teacher_entropy", t_entropy, prog_bar=True)
                self.log("student_entropy", s_entropy, prog_bar=True)
                self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss has collapsed: NaN or Inf")
            
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.update_teacher()

    # --------------------------------------------------------------------------- #
    #                           EMA & temperature helpers                         #
    # --------------------------------------------------------------------------- #
    def update_teacher(self):
        """EMA update after every optimizer.step()."""
        m = self.teacher_momentum()
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data * (1.0 - m))

    def teacher_momentum(self) -> float:
        return cosine_schedule(
            self.hp.momentum_base,
            self.hp.momentum_end,
            self.total_steps,
            self.global_step,
        )

    def teacher_temperature(self) -> float:
        # warm-up first `teacher_temp_warmup` steps linearly, then fixed
        if self.global_step < self.hp.teacher_temp_warmup:
            return (
                self.hp.teacher_temp_start
                + (self.hp.teacher_temp_end - self.hp.teacher_temp_start)
                * self.global_step
                / self.hp.teacher_temp_warmup
            )
        return self.hp.teacher_temp_end
    
    def compute_entropy(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()


    # ──────────────────────────────────────────────────────────────────────
    #  NEW: calculate the true step counts once the Trainer knows them
    # ──────────────────────────────────────────────────────────────────────
    def on_fit_start(self) -> None:
        dm      = self.trainer.datamodule
        dataset = dm.train_set                      # set in dm.setup()
        assert hasattr(dataset, "__len__"), "Dataset must implement __len__"

        # ------------------ 1) how many optimiser steps will run? ---------------
        # total samples that actually form batches (`drop_last=True` in DataLoader)
        global_batch = dm.batch_size * self.trainer.num_devices
        batches_per_epoch = len(dataset) // global_batch
        batches_per_epoch = max(1, batches_per_epoch)              # safety

        # account for gradient accumulation
        accum = int(self.trainer.accumulate_grad_batches)
        steps_per_epoch = math.ceil(batches_per_epoch / accum)

        self.total_steps  = steps_per_epoch * self.trainer.max_epochs
        self.warmup_steps = int(self.hparams.warmup_epochs / self.trainer.max_epochs * self.total_steps)

        # ------------------ 2) build the schedulers ------------------------------
        opt = self.optimizers(use_pl_optimizer=False)

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-5, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_steps - self.warmup_steps, eta_min=0.0
        )
        main_sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[self.warmup_steps]
        )

        wd_sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: (
                cosine_schedule(
                    self.hparams.weight_decay,
                    self.hparams.weight_decay_end,
                    self.total_steps,
                    step,
                ) / self.hparams.weight_decay
            ),
        )

        # ------------------ 3) register (mutate the list in-place!) -------------
        self.trainer.lr_scheduler_configs.extend(
            [
                LRSchedulerConfig(scheduler=main_sched, interval="step", name="lr"),
                LRSchedulerConfig(scheduler=wd_sched,   interval="step", name="wd"),
            ]
        )
    # helper used above
    def _wd_lambda(self, step: int):
        return cosine_schedule(
            self.hparams.weight_decay,
            self.hparams.weight_decay_end,
            self.total_steps,
            step,
        ) / self.hparams.weight_decay

    # --------------------------------------------------------------------------- #
    #                           optimizer & schedules                             #
    # --------------------------------------------------------------------------- #
    def configure_optimizers(self):
        # effective LR scaling -----------------------------------------------
        batch_size = self.trainer.datamodule.batch_size
        world      = max(1, self.trainer.num_devices)
        lr = self.hparams.base_lr * (batch_size * world) / 256
        print(f"Effective learning rate: {lr:.2e}")

        opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
            **getattr(self.hparams, "optimizer_kwargs", {}),
        )

        # return _only_ the optimiser; schedulers are registered later
        return opt
