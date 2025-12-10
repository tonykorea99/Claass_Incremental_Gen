"""
VAE 기반 Class Incremental Learning + Contrastive
- Conditional VAE 아님 (순수 VAE)
- μ 프로토타입 freeze 및 replay ratio 스케줄링 반영 버전
- 런타임 단축 모드
"""

# ============================ 1. Imports ============================
import random
import copy
import json
import datetime
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ============================ 2. Config ============================
@dataclass
class ExperimentConfig:
    seed: int = 42
    z_dim: int = 32
    base_k: int = 5
    epochs_base: int = 20
    epochs_cil: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    num_classes: int = 10

    # VAE loss
    beta_max: float = 0.8
    beta_warmup_base: int = 5
    beta_warmup_cil: int = 2
    kl_free_bits: float = 0.2

    # Contrastive
    use_contrastive_baseline: bool = False
    use_contrastive_exp: bool = True
    tau: float = 0.4
    lambda_con_max: float = 5e-3
    lambda_warmup_ep: int = 1
    hard_k: int = 5

    # Replay
    replay_ratio: float = 0.5
    replay_sigma: float = 0.5
    min_gen_per_cls: int = 8

    # System
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()

    # Metrics
    enable_metrics: bool = False
    eval_n_per_class: int = 1000
    eval_resize: int = 299
    metric_batch: int = 32

    # Paths & Logging
    out_root: str = "runs"
    save_plots: bool = True
    show_plots: bool = False
    save_sample_grid: bool = True
    sample_grid_nrow: int = 16
    save_model_each_step: bool = True

    # WandB
    wandb_entity: str = "hails"
    wandb_project: str = "Class_Incremental_Cons"
    wandb_group: str = "MNIST_CIL"
    wandb_tags: tuple = ("mnist", "vae", "cil", "gen-replay")
    log_batch_every: int = 100


CFG = ExperimentConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================ 3. Model ============================
class VAE(nn.Module):
    def __init__(self, zdim=CFG.z_dim, num_classes=CFG.num_classes):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        )
        self.enc_flat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_flat_dim, zdim)
        self.fc_lv = nn.Linear(self.enc_flat_dim, zdim)
        self.dec_fc = nn.Linear(zdim, self.enc_flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, x, y=None):
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        std = (0.5 * lv).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y=None):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 64, 7, 7)
        return self.dec_conv(h)

    def forward(self, x, y=None):
        mu, lv = self.encode(x, y)
        z = self.reparam(mu, lv)
        x_logits = self.decode(z, y)
        return x_logits, mu, lv, z

    @torch.no_grad()
    def embed_mu(self, x, y=None):
        mu, _ = self.encode(x, y)
        return mu


# ============================ 4. Utilities ============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_class_order(num_classes=10, base_k=CFG.base_k, seed=CFG.seed):
    r = random.Random(seed)
    classes = list(range(num_classes))
    r.shuffle(classes)
    return classes[:base_k], classes[base_k:]


def mnist_loader(selected, bs=CFG.batch_size, train=True, shuffle=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=train, download=True, transform=tfm)
    if selected is not None:
        idx = [i for i, (_, y) in enumerate(ds) if y in selected]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=CFG.num_workers)


def beta_at_epoch(ep, warmup_ep, beta_max=CFG.beta_max):
    if warmup_ep <= 0:
        return beta_max
    return beta_max * min(1.0, (ep + 1) / float(warmup_ep))


def kl_with_freebits(mu, logvar, free_bits=CFG.kl_free_bits):
    kl_elem = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
    if free_bits > 0.0:
        kl_elem = torch.clamp(kl_elem, min=free_bits)
    return kl_elem.sum(1).mean()


def vae_loss(x_logits, x, mu, lv, beta=1.0, free_bits=CFG.kl_free_bits):
    bce = F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")
    kl = kl_with_freebits(mu, lv, free_bits)
    total = bce + beta * kl
    return total, bce, kl


def lambda_con_at_epoch(ep, warmup_ep=CFG.lambda_warmup_ep, lam_max=CFG.lambda_con_max):
    if warmup_ep <= 0:
        return lam_max
    return lam_max * min(1.0, (ep + 1) / float(warmup_ep))


# ============================ 5. Base Trainer ============================
def train_base(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    model.train()
    for ep in range(CFG.epochs_base):
        beta = beta_at_epoch(ep, CFG.beta_warmup_base, CFG.beta_max)
        m_total = m_bce = m_kl = 0.0
        for x, y in tqdm(loader, desc=f"[BASE] {ep+1}/{CFG.epochs_base}", leave=False):
            x = x.to(DEVICE)
            logits, mu, lv, _ = model(x)
            total, bce, kl = vae_loss(logits, x, mu, lv, beta)
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            m_total += total.item()
            m_bce += bce.item()
            m_kl += kl.item()
        print(f"[BASE] ep {ep+1} | total={m_total/len(loader):.4f} | KL={m_kl/len(loader):.4f}")
    return model


# ============================ 6. CIL Trainer ============================
class CIL_Trainer:
    def __init__(self, model, base_classes, use_contrastive, label, run_dir, wb_run):
        self.model = model
        self.model.label = label
        self.use_contrastive = use_contrastive
        self.label = label
        self.run_dir = run_dir
        self.wb_run = wb_run
        self.opt = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)
        self.seen = base_classes.copy()
        self.global_batch_counter = 0
        self.metrics = defaultdict(list)
        self.current_new_c = None

        print(f"[{self.label}] Initializing prototype banks...")
        self.mu_proto_bank = self._update_mu_proto_frozen({}, base_classes)

    @torch.no_grad()
    def _update_mu_proto_frozen(self, mu_proto_bank, new_classes):
        loader = mnist_loader(new_classes, bs=256, train=False, shuffle=False)
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, _ = self.model.encode(x, y)
            for c in new_classes:
                mask = (y == c)
                if mask.any():
                    mu_c = mu[mask].mean(0)
                    mu_proto_bank[c] = mu_c.clone()
        return mu_proto_bank

    def _get_replay_ratio(self, step, total_steps):
        start_ratio, end_ratio = 0.5, 1.0
        return start_ratio + (end_ratio - start_ratio) * (step / max(1, total_steps - 1))

    @torch.no_grad()
    def _generate_replay(self, gen_model, classes, total_needed: int):
        if total_needed <= 0 or not classes:
            return None, None
        xs, ys = [], []
        per = max(CFG.min_gen_per_cls, total_needed // len(classes))
        for c in classes:
            mu = self.mu_proto_bank.get(c)
            if mu is None:
                continue
            z = mu.unsqueeze(0) + CFG.replay_sigma * torch.randn(per, mu.numel(), device=DEVICE)
            x_hat = torch.sigmoid(gen_model.decode(z))
            xs.append(x_hat)
            ys.append(torch.full((per,), c, dtype=torch.long, device=DEVICE))
        if not xs:
            return None, None
        X, Y = torch.cat(xs, 0), torch.cat(ys, 0)
        return X, Y

    def run_cil(self, stream_classes):
        total_steps = len(stream_classes)
        for step, new_c in enumerate(stream_classes, 1):
            self.current_new_c = new_c
            print(f"\n[{self.label}] === Step {step}/{total_steps}: learn class {new_c} ===")
            replay_ratio = self._get_replay_ratio(step, total_steps)
            print(f"Replay Ratio = {replay_ratio:.2f}")

            dl_new = mnist_loader([new_c], bs=CFG.batch_size, train=True)
            gen_model = copy.deepcopy(self.model).eval()

            for ep in range(CFG.epochs_cil):
                beta = beta_at_epoch(ep, CFG.beta_warmup_cil, CFG.beta_max)
                lam_eff = (
                    lambda_con_at_epoch(ep, CFG.lambda_warmup_ep, CFG.lambda_con_max)
                    / CFG.z_dim if self.use_contrastive else 0.0
                )
                m_total = m_bce = m_kl = 0.0
                for x_new, y_new in dl_new:
                    x_new, y_new = x_new.to(DEVICE), y_new.to(DEVICE)
                    need = int(x_new.size(0) * replay_ratio)
                    x_rep, y_rep = self._generate_replay(gen_model, self.seen, need)
                    if x_rep is not None:
                        x_all = torch.cat([x_new, x_rep], 0)
                        y_all = torch.cat([y_new, y_rep], 0)
                    else:
                        x_all, y_all = x_new, y_new
                    logits, mu, lv, _ = self.model(x_all, y_all)
                    total, bce, kl = vae_loss(logits, x_all, mu, lv, beta)
                    self.opt.zero_grad()
                    total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    m_total += total.item()
                    m_bce += bce.item()
                    m_kl += kl.item()

                print(f"[{self.label}] Step {step} Ep {ep+1} | BCE={m_bce/len(dl_new):.4f}")

            self.mu_proto_bank = self._update_mu_proto_frozen(self.mu_proto_bank, [new_c])
            self.seen.append(new_c)
        return self.metrics


# ============================ 7. Main ============================
def main():
    set_seed(CFG.seed)
    base_classes, stream_classes = make_class_order()
    print(f"Device: {DEVICE}")
    print(f"Base={base_classes}, Stream={stream_classes}")

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"VAE_{time_str}_freeze_replay"
    run_dir = Path(CFG.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Base Pretrain
    print("\n--- Base Model Training ---")
    base_loader = mnist_loader(base_classes, bs=CFG.batch_size, train=True)
    base_model = VAE(CFG.z_dim, CFG.num_classes).to(DEVICE)
    base_model = train_base(base_model, base_loader)
    torch.save(base_model.state_dict(), run_dir / "base_model.pt")

    # Contrastive (CL 적용 버전)
    print("\n=== CIL (Contrastive + Freeze + Replay Schedule) ===")
    modelB = copy.deepcopy(base_model)
    wb_runB = wandb.init(
        entity=CFG.wandb_entity,
        project=CFG.wandb_project,
        group=CFG.wandb_group,
        tags=list(CFG.wandb_tags) + ["contrastive"],
        name=run_name + "_Contrastive",
        dir=str(run_dir),
        reinit=True,
    )
    trainerB = CIL_Trainer(
        modelB, base_classes, use_contrastive=CFG.use_contrastive_exp,
        label="B_Contrastive", run_dir=run_dir, wb_run=wb_runB
    )
    trainerB.run_cil(stream_classes)
    wb_runB.finish()

    print(f"\nAll done. Results saved to: {run_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
