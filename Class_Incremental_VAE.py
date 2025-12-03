"""
Class Incremental Learning with Conditional VAE (v10.5: Attribute Error Fixed).

- Fix: Ensured '_generate_samples_for_class' is correctly indented inside 'MetricCalculator'.
- Logic: v10.4 (Ultra-low Lambda, High Epochs, Visual Proofs).
- Logging: Summary/Per_Class/Train split.
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

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# [Auto Login]
os.environ["WANDB_API_KEY"] = "e758b93c3e805dafd9d187ec1c0b1b984fe6256f"

# Global Configs
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")
torch.backends.cudnn.benchmark = True

# ============================ 2. Configuration ============================
@dataclass
class ExperimentConfig:
    """Configuration with Final Corrected Hyperparameters."""
    # --- Data & Model ---
    seed: int = 42
    z_dim: int = 32
    base_k: int = 5
    epochs_base: int = 20
    epochs_cil: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    num_classes: int = 10

    # --- VAE Loss ---
    beta_max: float = 0.8
    beta_warmup_base: int = 5
    beta_warmup_cil: int = 2
    kl_free_bits: float = 0.2

    # --- Contrastive Loss ---
    use_contrastive_baseline: bool = False
    use_contrastive_exp: bool = True
    tau: float = 0.4
    lambda_con_max: float = 0.000001 # 1e-6
    lambda_warmup_ep: int = 1
    hard_k: int = 5

    # --- Generative Replay ---
    replay_ratio: float = 0.5
    replay_sigma: float = 1.0
    min_gen_per_cls: int = 8

    # --- System ---
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()

    # --- Evaluation ---
    eval_n_per_class: int = 1000
    eval_resize: int = 299
    eval_every_step: bool = True
    metric_batch: int = 32
    lpips_backbone: str = "alex"
    lpips_resize: int = 224

    # --- Logging ---
    out_root: str = "runs"
    save_plots: bool = True
    show_plots: bool = False
    save_sample_grid: bool = True
    sample_grid_nrow: int = 16

    # --- W&B Settings ---
    wandb_entity: str = "hails"
    wandb_project: str = "Class_Incremental_Cons"
    wandb_group: str = "MNIST_CIL"
    wandb_tags: tuple = ("mnist", "cvae", "cil", "gen-replay", "metrics", "fixed")

CFG = ExperimentConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================ 3. Model Definition ============================
class ConditionalVAE(nn.Module):
    """Conditional VAE for MNIST."""
    def __init__(self, zdim=CFG.z_dim, num_classes=CFG.num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 10)

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.enc_flat_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_flat_dim + 10, zdim)
        self.fc_lv = nn.Linear(self.enc_flat_dim + 10, zdim)

        self.dec_fc = nn.Linear(zdim + 10, self.enc_flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(),
            nn.Linear(zdim, zdim)
        )

    def encode(self, x, y):
        h = self.enc_conv(x).flatten(1)
        y_emb = self.label_emb(y)
        h_cat = torch.cat([h, y_emb], dim=1)
        return self.fc_mu(h_cat), self.fc_lv(h_cat)

    def reparam(self, mu, lv):
        std = (0.5 * lv).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_emb(y)
        z_cat = torch.cat([z, y_emb], dim=1)
        h = self.dec_fc(z_cat)
        h = h.reshape(h.size(0), 64, 7, 7)
        return self.dec_conv(h)

    def forward(self, x, y):
        mu, lv = self.encode(x, y)
        z = self.reparam(mu, lv)
        x_logits = self.decode(z, y)
        return x_logits, mu, lv, z

    @torch.no_grad()
    def embed_mu(self, x, y, project=True):
        mu, _ = self.encode(x, y)
        return self.proj(mu) if project else mu

# ============================ 4. Utilities ============================
def require_generation_metrics():
    _ = (FrechetInceptionDistance, InceptionScore,
         LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_class_order(num_classes=10, base_k=CFG.base_k, seed=CFG.seed):
    r = random.Random(seed); classes = list(range(num_classes)); r.shuffle(classes)
    return classes[:base_k], classes[base_k:]

def mnist_loader(selected, bs=CFG.batch_size, train=True, shuffle=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=train, download=True, transform=tfm)
    if selected is not None:
        idx = [i for i, (_, y) in enumerate(ds) if y in selected]
        sub = Subset(ds, idx)
    else: sub = ds
    return DataLoader(sub, batch_size=bs, shuffle=shuffle,
                      num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

def beta_at_epoch(ep, warmup_ep, beta_max=CFG.beta_max):
    if warmup_ep <= 0: return beta_max
    return beta_max * min(1.0, (ep + 1) / float(warmup_ep))

def kl_with_freebits(mu, logvar, free_bits=CFG.kl_free_bits):
    kl_elem = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
    if free_bits > 0.0: kl_elem = torch.clamp(kl_elem, min=free_bits)
    return kl_elem.sum(1).mean()

def vae_loss(x_logits, x, mu, lv, beta=1.0, free_bits=CFG.kl_free_bits):
    bce = F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")
    kl = kl_with_freebits(mu, lv, free_bits)
    total = bce + beta * kl
    return total, bce, kl

def lambda_con_at_epoch(ep, warmup_ep=CFG.lambda_warmup_ep, lam_max=CFG.lambda_con_max):
    if warmup_ep <= 0: return lam_max
    return lam_max * min(1.0, (ep + 1) / float(warmup_ep))

def info_nce(pos_logit, neg_logits):
    logits = torch.cat([pos_logit, neg_logits], 1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def optimize_model_for_runtime(model: nn.Module) -> nn.Module:
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return model

@torch.no_grad()
def _to_3ch_and_resize(x: torch.Tensor, size: int):
    if x.shape[1] == 3:
        if x.shape[2] == size and x.shape[3] == size: return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x3 = x.repeat(1, 3, 1, 1)
    return F.interpolate(x3, size=(size, size), mode="bilinear", align_corners=False)

@torch.no_grad()
def save_image_grid(img_tensor, path, nrow=CFG.sample_grid_nrow):
    grid = make_grid(img_tensor, nrow=nrow, padding=2)
    save_image(grid, path)

def plot_metric(metricsA, metricsB, key, title, ylabel, save_dir, fname):
    plt.figure(figsize=(10, 6))
    if key in metricsA: plt.plot(metricsA["steps"], metricsA[key], label="Baseline (A)", marker='o')
    if key in metricsB: plt.plot(metricsB["steps"], metricsB[key], label="Contrastive (B)", marker='x')
    plt.title(title); plt.xlabel("CIL Step"); plt.ylabel(ylabel); plt.legend()
    plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / fname)
    if CFG.show_plots: plt.show()
    plt.close()

# ============================ 5. Evaluation Functions ============================
@torch.no_grad()
def generate_eval_samples(model, classes, per_class: int, mu_proto_bank: dict):
    # Global function for logging samples
    real_loader = mnist_loader(classes, bs=256, train=False, shuffle=True)
    real = []
    for x, y in real_loader:
        if all(c.item() in classes for c in y): real.append(x)
        if sum(t.size(0) for t in real) >= per_class * len(classes): break
    
    X_real = torch.cat(real, 0)[:per_class * len(classes)].to(DEVICE)
    xs = []
    for c in classes:
        if c not in mu_proto_bank: continue
        y_c = torch.full((per_class,), c, dtype=torch.long, device=DEVICE)
        mu_c = mu_proto_bank[c].to(DEVICE)
        z = mu_c.unsqueeze(0) + CFG.replay_sigma * torch.randn(per_class, mu_c.numel(), device=DEVICE)
        xs.append(torch.sigmoid(model.decode(z, y_c)))

    X_fake = torch.cat(xs, 0) if xs else torch.empty(0, 1, 28, 28, device=DEVICE)
    return X_real, X_fake

@torch.no_grad()
def analyze_proto_separation(classes, proto_bank: dict):
    vecs = [F.normalize(proto_bank[c].to(DEVICE), dim=0) for c in classes if c in proto_bank]
    if len(vecs) < 2: return {}
    V = torch.stack(vecs, 0)
    cos_sim = V @ V.t()
    eu_dist = torch.cdist(V, V, p=2)
    mask = ~torch.eye(V.size(0), dtype=torch.bool, device=V.device)
    return {
        "cos_mean": float(cos_sim[mask].mean().item()),
        "cos_min": float(cos_sim[mask].min().item()),
        "eu_mean": float(eu_dist[mask].mean().item()),
        "eu_min": float(eu_dist[mask].min().item())
    }

# ============================ 6. Base Trainer ============================
def train_base(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    model.train()
    for ep in range(CFG.epochs_base):
        beta = beta_at_epoch(ep, CFG.beta_warmup_base, CFG.beta_max)
        m_total = m_bce = m_kl = 0.0
        for x, y in tqdm(loader, desc=f"[BASE] {ep+1}/{CFG.epochs_base}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if DEVICE == "cuda": x = x.to(memory_format=torch.channels_last)
            logits, mu, lv, _ = model(x, y)
            total, bce, kl = vae_loss(logits, x, mu, lv, beta, CFG.kl_free_bits)
            opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            m_total += total.item(); m_bce += bce.item(); m_kl += kl.item()
        n = len(loader)
        print(f"[BASE] ep {ep+1} | beta={beta:.2f} | total={m_total/n:.4f} | BCE={m_bce/n:.4f} | KL={m_kl/n:.4f}")
    return model

# ============================ 7. Metric Calculator ============================
class MetricCalculator:
    def __init__(self, model, classes_to_eval, mu_proto_bank, proto_bank):
        self.model = model; self.classes_to_eval = classes_to_eval
        self.mu_proto_bank = mu_proto_bank; self.proto_bank = proto_bank
        self.metrics = {}
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(DEVICE)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=CFG.lpips_backbone, reduction=None).to(DEVICE)
        self.global_fid = FrechetInceptionDistance().to(DEVICE)
        self.global_isc = InceptionScore().to(DEVICE)
        self.per_class_fid = FrechetInceptionDistance().to(DEVICE)
        self.per_class_isc = InceptionScore().to(DEVICE)

    def _log(self, name, value): self.metrics[name] = value

    @torch.no_grad()
    def run_all_metrics(self) -> dict:
        self.model.eval()
        self._calculate_reconstruction_metrics()
        self._calculate_generation_metrics()
        self._calculate_separation_metrics()
        self.model.train()
        return self.metrics

    @torch.no_grad()
    def _calculate_reconstruction_metrics(self):
        test_loader = mnist_loader(selected=self.classes_to_eval, bs=CFG.metric_batch, train=False, shuffle=False)
        all_ssim, all_lpips, all_y = [], [], []
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_logits, _, _, _ = self.model(x, y)
            x_recon = torch.sigmoid(x_logits)
            
            ssim_vals = self.ssim_metric(x_recon, x)
            if ssim_vals.ndim == 0: ssim_vals = ssim_vals.unsqueeze(0)
            
            x_recon_lp = (_to_3ch_and_resize(x_recon, CFG.lpips_resize) * 2.0) - 1.0
            x_real_lp = (_to_3ch_and_resize(x, CFG.lpips_resize) * 2.0) - 1.0
            lpips_vals = self.lpips_metric(x_recon_lp, x_real_lp)
            if lpips_vals.ndim == 0: lpips_vals = lpips_vals.unsqueeze(0)
            
            all_ssim.append(ssim_vals.cpu()); all_lpips.append(lpips_vals.cpu()); all_y.append(y.cpu())

        if not all_y: return
        all_ssim, all_lpips, all_y = torch.cat(all_ssim), torch.cat(all_lpips), torch.cat(all_y)
        
        for c in self.classes_to_eval:
            mask = (all_y == c)
            if mask.any():
                self._log(f"Per_Class/Recon/cl_{c}_SSIM", all_ssim[mask].mean().item())
                self._log(f"Per_Class/Recon/cl_{c}_LPIPS", all_lpips[mask].mean().item())
                
        self._log("Summary/Recon_SSIM", all_ssim.mean().item())
        self._log("Summary/Recon_LPIPS", all_lpips.mean().item())

    @torch.no_grad()
    def _calculate_generation_metrics(self):
        self.global_fid.reset(); self.global_isc.reset()
        bs = CFG.metric_batch
        for c in self.classes_to_eval:
            # [FIXED] Call internal method self._generate_samples_for_class
            real_c, fake_c = self._generate_samples_for_class(c, CFG.eval_n_per_class)
            if real_c is None or fake_c is None: continue
            
            real_c_u8 = (_to_3ch_and_resize(real_c, CFG.eval_resize) * 255).clamp(0, 255).to(torch.uint8)
            fake_c_u8 = (_to_3ch_and_resize(fake_c, CFG.eval_resize) * 255).clamp(0, 255).to(torch.uint8)
            
            self.per_class_fid.reset(); self.per_class_isc.reset()
            # pylint: disable=too-many-function-args
            for i in range(0, real_c_u8.size(0), bs):
                self.per_class_fid.update(real_c_u8[i:i+bs], real=True)
                self.global_fid.update(real_c_u8[i:i+bs], real=True)
            for i in range(0, fake_c_u8.size(0), bs):
                self.per_class_fid.update(fake_c_u8[i:i+bs], real=False)
                self.per_class_isc.update(fake_c_u8[i:i+bs])
                self.global_fid.update(fake_c_u8[i:i+bs], real=False)
                self.global_isc.update(fake_c_u8[i:i+bs])
            
            self._log(f"Per_Class/Gen/cl_{c}_FID", self.per_class_fid.compute().item())
            val_is, _ = self.per_class_isc.compute()
            self._log(f"Per_Class/Gen/cl_{c}_IS", val_is.item())

        try:
            self._log("Summary/Gen_FID", self.global_fid.compute().item())
            val_is_all, _ = self.global_isc.compute()
            self._log("Summary/Gen_IS", val_is_all.item())
        # pylint: disable=broad-exception-caught
        except Exception as e: print(f"Warning: Global FID/IS Error: {e}")

    # [FIXED] This method is now correctly indented inside the class
    @torch.no_grad()
    def _generate_samples_for_class(self, c: int, n: int):
        real_loader_c = mnist_loader(selected=[c], bs=n, train=False, shuffle=True)
        try: X_real_c = next(iter(real_loader_c))[0].to(DEVICE)
        except StopIteration: return None, None
        if c not in self.mu_proto_bank: return X_real_c, None
        mu_c = self.mu_proto_bank[c].to(DEVICE)
        z = mu_c.unsqueeze(0) + CFG.replay_sigma * torch.randn(n, mu_c.numel(), device=DEVICE)
        y_c = torch.full((n,), c, dtype=torch.long, device=DEVICE)
        X_fake_c = torch.sigmoid(self.model.decode(z, y_c))
        return X_real_c, X_fake_c

    @torch.no_grad()
    def _calculate_separation_metrics(self):
        sep_metrics = analyze_proto_separation(self.classes_to_eval, self.proto_bank)
        for k, v in sep_metrics.items(): self._log(f"Summary/Sep_{k}", v)

# ============================ 8. CIL Trainer ============================
class CIL_Trainer:
    def __init__(self, model, base_classes, use_contrastive, label, run_dir, wb_run):
        self.model = model; self.model.label = label
        self.use_contrastive = use_contrastive; self.label = label
        self.run_dir = run_dir; self.wb_run = wb_run
        self.opt = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)
        self.seen = base_classes.copy()
        self.global_batch_counter = 0; self.metrics = defaultdict(list); self.current_new_c = None
        print(f"[{self.label}] Initializing prototype banks...")
        self.proto_bank = self._update_proto({}, self.seen)
        self.mu_proto_bank = self._update_mu_proto({}, self.seen)
        
    @torch.no_grad()
    def _update_proto(self, proto_bank, classes, ema=0.9):
        if not classes: return proto_bank
        loader = mnist_loader(classes, bs=256, train=False, shuffle=False)
        sums, counts = {c: None for c in classes}, {c: 0 for c in classes}
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = self.model.embed_mu(x, y, project=True)
            for c in classes:
                m = (y == c)
                if m.any():
                    add = z[m].sum(0); sums[c] = add if sums[c] is None else sums[c] + add
                    counts[c] += int(m.sum().item())
        for c in classes:
            if counts[c] == 0: continue
            new = sums[c] / counts[c]
            proto_bank[c] = (ema * proto_bank[c] + (1 - ema) * new) if c in proto_bank else new.clone()
        return proto_bank

    @torch.no_grad()
    def _update_mu_proto(self, mu_proto_bank, classes, ema=0.9):
        if not classes: return mu_proto_bank
        loader = mnist_loader(classes, bs=256, train=False, shuffle=False)
        sums, counts = {c: None for c in classes}, {c: 0 for c in classes}
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mu, _ = self.model.encode(x, y)
            for c in classes:
                m = (y == c)
                if m.any():
                    add = mu[m].sum(0); sums[c] = add if sums[c] is None else sums[c] + add
                    counts[c] += int(m.sum().item())
        for c in classes:
            if counts[c] == 0: continue
            new = sums[c] / counts[c]
            mu_proto_bank[c] = (ema * mu_proto_bank[c] + (1 - ema) * new) if c in mu_proto_bank else new.clone()
        return mu_proto_bank
        
    @torch.no_grad()
    def _generate_replay(self, gen_model, classes, total_needed: int):
        if total_needed <= 0 or not classes: return None, None
        xs, ys = [], []
        per = max(CFG.min_gen_per_cls, total_needed // max(1, len(classes)))
        for c in classes:
            if c not in self.mu_proto_bank: continue
            mu_c = self.mu_proto_bank[c].to(DEVICE)
            z = mu_c.unsqueeze(0) + CFG.replay_sigma * torch.randn(per, mu_c.numel(), device=DEVICE)
            y_c = torch.full((per,), c, dtype=torch.long, device=DEVICE)
            x_hat = torch.sigmoid(gen_model.decode(z, y_c))
            xs.append(x_hat); ys.append(y_c)
        if not xs: return None, None
        X, Y = torch.cat(xs, 0), torch.cat(ys, 0)
        if X.size(0) > total_needed:
            idx = torch.randperm(X.size(0), device=DEVICE)[:total_needed]; X, Y = X[idx], Y[idx]
        return X, Y

    @torch.no_grad()
    def _build_pos_neg(self, z_new, y, seen, hard_k):
        _, D = z_new.shape
        z_pos = torch.empty_like(z_new)
        for c in y.unique():
            mask = (y == c)
            cls_vecs = z_new[mask]
            if cls_vecs.size(0) >= 2: m = cls_vecs.mean(0)
            elif int(c.item()) in self.proto_bank: m = self.proto_bank[int(c.item())].to(z_new.device)
            else: m = cls_vecs[0]
            z_pos[mask] = m
        if any(c in self.proto_bank for c in seen):
            neg_mat = torch.stack([self.proto_bank[c] for c in seen if c in self.proto_bank], dim=0).to(z_new.device)
        else: neg_mat = torch.zeros(1, D, device=z_new.device)
        bmean = F.normalize(z_new.mean(0, keepdim=True), dim=1)
        negn = F.normalize(neg_mat, dim=1)
        # pylint: disable=not-callable
        sims = F.cosine_similarity(bmean, negn)
        k = min(hard_k, neg_mat.size(0))
        _, idx = sims.topk(k)
        z_neg = neg_mat[idx]
        return z_pos, z_neg

    def _calculate_batch_loss_corrected(self, logits, x_all, y_all, mu, lv, beta, lam):
        total, bce, kl = vae_loss(logits, x_all, mu, lv, beta, CFG.kl_free_bits)
        batch_p_pos, con_loss_val = 0.0, 0.0
        if self.use_contrastive and self.seen:
            z = F.normalize(self.model.proj(mu), dim=1)
            z_pos, z_neg = self._build_pos_neg(z, y_all, self.seen, CFG.hard_k)
            z_pos = F.normalize(z_pos, dim=1); z_neg = F.normalize(z_neg, dim=1)
            pos_logit = (z * z_pos).sum(1, keepdim=True) / CFG.tau
            neg_logits = (z @ z_neg.t()) / CFG.tau if z_neg.numel() else torch.zeros(z.size(0), 1, device=z.device)
            con_loss = info_nce(pos_logit, neg_logits)
            con_loss_val = con_loss.item()
            total = total + lam * con_loss
            batch_p_pos = float(torch.softmax(torch.cat([pos_logit, neg_logits], 1), 1)[:, 0].mean().item())
        return total, bce, kl, con_loss_val, batch_p_pos

    def _log_batch_to_wandb(self, total, bce, kl, con_loss_val, batch_p_pos, x_all, y_all):
        wb_payload = {"Train/total": total.item(), "Train/BCE": bce.item(), "Train/KL": kl.item()}
        if self.use_contrastive:
            wb_payload["Train/p_pos"] = batch_p_pos
            wb_payload["Train/con_loss"] = con_loss_val
        
        # [Visual Proof] Log inputs occasionally
        if self.global_batch_counter % 50 == 0:
            grid = make_grid(x_all[:64], nrow=8, padding=2)
            wb_payload["Train/Inputs/Mix_Batch"] = wandb.Image(grid, caption=f"Batch (Labels: {y_all[:64].tolist()})")

        self.wb_run.log(wb_payload, step=self.global_batch_counter)
        
    @torch.no_grad()
    def _run_epoch_evaluation(self, classes_to_eval) -> dict:
        self.proto_bank = self._update_proto(self.proto_bank, classes_to_eval, ema=0.9)
        self.mu_proto_bank = self._update_mu_proto(self.mu_proto_bank, classes_to_eval, ema=0.9)
        calculator = MetricCalculator(self.model, classes_to_eval, self.mu_proto_bank, self.proto_bank)
        return calculator.run_all_metrics()

    def _log_epoch_to_wandb(self, epoch_metrics: dict, eval_metrics: dict):
        wb_payload = {
            "Train/Summary/Total": epoch_metrics["total"], "Train/Summary/BCE": epoch_metrics["bce"],
            "Train/Summary/KL": epoch_metrics["kl"]
        }
        if self.use_contrastive: wb_payload["Train/Summary/P_Pos"] = epoch_metrics["p_pos"]
        wb_payload.update(eval_metrics)
        
        if CFG.save_sample_grid:
            with torch.no_grad():
                classes_to_sample = self.seen + [self.current_new_c]
                X_real_small, X_fake_small = generate_eval_samples(self.model, classes_to_sample, 8, self.mu_proto_bank)
                if X_real_small.numel() > 0:
                    wb_payload["Samples/Real_Grid"] = wandb.Image(make_grid(X_real_small[:64], nrow=8, padding=2))
                if X_fake_small.numel() > 0:
                    wb_payload["Samples/Fake_Grid"] = wandb.Image(make_grid(X_fake_small[:64], nrow=8, padding=2))
        
        self.wb_run.log(wb_payload, step=self.global_batch_counter)
        
    def _save_local_artifacts(self, step, eval_metrics: dict):
        if self.run_dir is None: return
        step_log = {"step": step, "label": self.label, **eval_metrics}
        with open(self.run_dir / "gen_metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step_log) + "\n")
        if CFG.save_sample_grid:
            X_real_small, X_fake_small = generate_eval_samples(self.model, self.seen, 16, self.mu_proto_bank)
            save_image_grid(X_real_small[:CFG.sample_grid_nrow * CFG.sample_grid_nrow], self.run_dir / f"{self.label}_step{step:02d}_real.png")
            save_image_grid(X_fake_small[:CFG.sample_grid_nrow * CFG.sample_grid_nrow], self.run_dir / f"{self.label}_step{step:02d}_fake.png")

    def run_cil(self, stream_classes):
        for step, new_c in enumerate(stream_classes, 1):
            self.current_new_c = new_c
            print(f"\n[{self.label}] === Step {step} learn {new_c} ===")
            dl_new = mnist_loader([new_c], bs=CFG.batch_size, train=True, shuffle=True)
            self.model.train()
            gen_model = copy.deepcopy(self.model).eval() 
            for p in gen_model.parameters(): p.requires_grad_(False)
            
            for ep in range(CFG.epochs_cil):
                beta = beta_at_epoch(ep, CFG.beta_warmup_cil, CFG.beta_max)
                lam = lambda_con_at_epoch(ep, CFG.lambda_warmup_ep, CFG.lambda_con_max) if self.use_contrastive else 0.0
                m_total = m_bce = m_kl = m_ppos = 0.0
                n_batches = 0
                for x_new, y_new in dl_new:
                    x_new, y_new = x_new.to(DEVICE), y_new.to(DEVICE)
                    if DEVICE == "cuda": x_new = x_new.to(memory_format=torch.channels_last)
                    need = int(x_new.size(0) * CFG.replay_ratio)
                    x_rep, y_rep = self._generate_replay(gen_model, self.seen, need)
                    if x_rep is not None: 
                        x_all = torch.cat([x_new, x_rep], 0)
                        y_all = torch.cat([y_new, y_rep], 0)
                        shuffler = torch.randperm(x_all.size(0))
                        x_all = x_all[shuffler]
                        y_all = y_all[shuffler]
                    else: x_all, y_all = x_new, y_new
                    logits, mu, lv, _ = self.model(x_all, y_all)
                    total, bce, kl, con_loss_val, batch_p_pos = self._calculate_batch_loss_corrected(logits, x_all, y_all, mu, lv, beta, lam)
                    self.opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.opt.step()
                    m_total += total.item(); m_bce += bce.item(); m_kl += kl.item(); m_ppos += batch_p_pos; n_batches += 1
                    self.global_batch_counter += 1
                    if self.wb_run: 
                        self._log_batch_to_wandb(total, bce, kl, con_loss_val, batch_p_pos, x_all, y_all)
                
                epoch_metrics = {"total": m_total / n_batches, "bce": m_bce / n_batches, "kl": m_kl / n_batches, "p_pos": m_ppos / max(1, n_batches)}
                classes_to_eval = self.seen + [new_c]
                eval_metrics = self._run_epoch_evaluation(classes_to_eval)
                self.metrics["steps"].append(step); self.metrics["global_batch"].append(self.global_batch_counter)
                for k, v in epoch_metrics.items(): self.metrics[k].append(v)
                for k, v in eval_metrics.items(): 
                    safe_k = k.replace("/", "_")
                    self.metrics[safe_k].append(v)

                if self.wb_run: self._log_epoch_to_wandb(epoch_metrics, eval_metrics)
                gm = eval_metrics.get("Summary/Gen_FID", 0.0)
                print(f"[{self.label}] step {step} ep {ep+1} | BCE Avg: {epoch_metrics['bce']:.4f} | Global FID: {gm:.3f}")

            self.seen.append(new_c); self._save_local_artifacts(step, eval_metrics)
            self.proto_bank = self._update_proto(self.proto_bank, self.seen)
            self.mu_proto_bank = self._update_mu_proto(self.mu_proto_bank, self.seen)
            print(f"[{self.label}] step {step} finished.")
        return self.metrics

# ============================ 9. Main Execution Block ============================
def main():
    require_generation_metrics()
    set_seed(CFG.seed)
    
    base_classes, stream_classes = make_class_order()
    print(f"Device: {DEVICE}"); print(f"[Class order] base={base_classes}, stream={stream_classes}")
    
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"CVAE_{time_str}_z{CFG.z_dim}_L{CFG.lambda_con_max}"
    run_dir = Path(CFG.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    runA = run_dir / "A_Baseline"; runB = run_dir / "B_Contrastive"
    runA.mkdir(exist_ok=True); runB.mkdir(exist_ok=True)
    
    with open(run_dir / "config.json", "w", encoding="utf-8") as f: json.dump(asdict(CFG), f, indent=4)
    
    print("\n--- Base Model Training (Pre-train Generator) ---")
    base_model = optimize_model_for_runtime(ConditionalVAE(CFG.z_dim).to(DEVICE))
    base_loader = mnist_loader(base_classes, CFG.batch_size, True, True)
    base_model = train_base(base_model, base_loader)
    
    # Base Verification Log
    print("\n--- Verifying Base Model Generator Capability ---")
    base_model.eval()
    
    print("\n--- Experiment A (Baseline) ---")
    modelA = copy.deepcopy(base_model)
    wbA = wandb.init(
        entity=CFG.wandb_entity, project=CFG.wandb_project, group=CFG.wandb_group,
        name=f"{run_name}_Baseline", config=asdict(CFG), tags=CFG.wandb_tags, reinit=True
    )
    trainerA = CIL_Trainer(modelA, base_classes, False, "A", runA, wbA)
    wbA.log({"Samples/Fake_Grid": wandb.Image(make_grid(generate_eval_samples(modelA, base_classes, 16, trainerA.mu_proto_bank)[1], nrow=8, padding=2), caption="Initial Base Generator")}, step=0)
    metricsA = trainerA.run_cil(stream_classes)
    wbA.finish()
    print("--- Experiment A (Baseline) Finished ---")

    print("\n--- Experiment B (Contrastive Replay) ---")
    modelB = copy.deepcopy(base_model)
    wbB = wandb.init(
        entity=CFG.wandb_entity, project=CFG.wandb_project, group=CFG.wandb_group,
        name=f"{run_name}_Contrastive", config=asdict(CFG), tags=CFG.wandb_tags, reinit=True
    )
    trainerB = CIL_Trainer(modelB, base_classes, True, "B", runB, wbB)
    wbB.log({"Samples/Fake_Grid": wandb.Image(make_grid(generate_eval_samples(modelB, base_classes, 16, trainerB.mu_proto_bank)[1], nrow=8, padding=2), caption="Initial Base Generator")}, step=0)
    metricsB = trainerB.run_cil(stream_classes)
    wbB.finish()
    print("--- Experiment B (Contrastive) Finished ---")

    if CFG.save_plots:
        print("\n--- Saving local comparison plots ---")
        plot_metric(metricsA, metricsB, "Summary/Recon_SSIM", "Recon SSIM (All)", "SSIM", save_dir=runB, fname="plot_ssim_recon.png")
        plot_metric(metricsA, metricsB, "Summary/Gen_FID", "Gen FID (All)", "FID", save_dir=runB, fname="plot_fid_gen.png")

if __name__ == "__main__":
    mp.freeze_support()
    if os.name == 'posix':
        try: mp.set_start_method("spawn", force=True)
        except RuntimeError: pass
    main()