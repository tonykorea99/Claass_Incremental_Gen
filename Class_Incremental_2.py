# ============================
# v8.3: OOM Hotfix
# ============================

# ============================ 1. Imports ============================
from dataclasses import dataclass
import os, random, copy, json, datetime, numpy as np, multiprocessing as mp
from collections import defaultdict
from pathlib import Path

# --- PyTorch & TorchVision ---
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# --- Plotting & Utilities ---
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Core Metrics ---
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

import warnings
import wandb # Weights & Biases 로깅

# --- Global Configs ---
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")
torch.backends.cudnn.benchmark = True # CUDA 연산 가속화

# ============================ 2. Configuration ============================
@dataclass
class CFG:
    """
    실험의 모든 하이퍼파라미터와 설정을 관리하는 중앙 클래스입니다.
    """
    # --- Data & Model ---
    seed:int = 42
    z_dim:int = 32         # VAE의 잠재 공간(Latent Space) 차원
    base_k:int = 5         # 초기 Base Task에서 학습할 클래스 수
    epochs_base:int = 5    # Base Task 학습 에폭 수
    epochs_cil:int  = 3    # Incremental Task 학습 에폭 수
    batch_size:int  = 128
    lr:float = 1e-3        # 학습률

    # --- VAE Loss (KL Divergence) ---
    beta_max:float = 1.0   # KL 손실의 최대 가중치 (beta-VAE)
    beta_warmup_base:int = 2 # Base 학습 시 beta 워밍업 에폭
    beta_warmup_cil:int  = 1 # CIL 학습 시 beta 워밍업 에폭
    kl_free_bits:float   = 0.2 # KL 손실 계산 시 최소 nats (Free Bits)

    # --- Contrastive Loss ---
    use_contrastive_baseline:bool = False # A 모델(Baseline)은 Contrastive 비활성화
    use_contrastive_exp:bool      = True  # B 모델(실험군)은 Contrastive 활성화
    tau:float = 0.4                       # Contrastive Loss의 온도(temperature) 파라미터
    lambda_con_max:float = 0.05           # Contrastive 손실의 최대 가중치 (v6 설정)
    lambda_warmup_ep:int = 1              # Lambda 워밍업 에폭
    hard_k:int = 5                        # Hard Negative Mining을 위한 K값

    # --- Generative Replay ---
    replay_ratio:float = 0.5   # 새 데이터 대비 리플레이 데이터의 비율
    replay_sigma:float = 1.0   # 리플레이 생성 시 mu에 가해지는 노이즈 표준편차
    min_gen_per_cls:int = 8    # 리플레이 시 클래스당 최소 생성 샘플 수

    # --- System & Loader ---
    num_workers:int = 0  # DataLoader 워커 수 (Windows 호환성을 위해 0)
    pin_memory:bool = torch.cuda.is_available()

    # --- Evaluation Metrics ---
    # [피드백 1.1] N 개수 하이퍼파라미터화 및 증가
    eval_n_per_class:int = 1000   # (v7: 200) 평가 시 클래스당 샘플 수 (MNIST Testset 클래스당 1000개)
    eval_resize:int = 299        # FID/IS 계산을 위한 Inception 모델 입력 크기
    eval_every_step:bool = True  # CIL 스텝(에폭)마다 평가 수행 여부
    metric_batch:int = 32        # [v8.3] OOM 방지를 위해 평가 배치 크기 줄임
    lpips_backbone:str = "alex"  # LPIPS 계산 시 사용할 백본
    lpips_resize:int = 224       # LPIPS 계산을 위한 이미지 크기

    # --- Logging & Saving ---
    out_root: str = "runs"       # 로컬 결과물 저장 루트 디렉토리
    save_plots: bool = True      # 로컬에 비교 플롯 저장 여부
    show_plots: bool = False     # 스크립트 실행 시 플롯 표시 여부
    save_sample_grid: bool = True # 생성된 샘플 이미지 그리드 저장 여부
    sample_grid_nrow: int = 16   # 샘플 그리드의 행당 이미지 수

    # --- W&B Project ---
    wandb_project:str = "MNIST-CILv9_OOM_fix" # [수정] 새 프로젝트 이름
    wandb_group:str = "MNIST_CIL"     # 두 실험(A, B)을 묶을 그룹 이름
    wandb_tags:tuple = ("mnist","vae","cil","gen-replay","metrics")

CFG = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================ 3. Model Definition ============================
class VAE(nn.Module):
    def __init__(self, zdim=CFG.z_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64*7*7, zdim)
        self.fc_lv = nn.Linear(64*7*7, zdim)
        self.fc    = nn.Linear(zdim, 64*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(zdim, zdim), 
            nn.ReLU(), 
            nn.Linear(zdim, zdim)
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        std = (0.5*lv).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc(z).reshape(z.size(0), 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        x_logits = self.decode(z)
        return x_logits, mu, lv, z

    @torch.no_grad()
    def embed_mu(self, x, project=True):
        mu, _ = self.encode(x)
        return self.proj(mu) if project else mu

# ============================ 4. Stateless Utilities ============================

def require_generation_metrics():
    _ = FrechetInceptionDistance; _ = InceptionScore
    _ = LearnedPerceptualImagePatchSimilarity; _ = StructuralSimilarityIndexMeasure

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_class_order(num_classes=10, base_k=CFG.base_k, seed=CFG.seed):
    r = random.Random(seed); classes = list(range(num_classes)); r.shuffle(classes)
    return classes[:base_k], classes[base_k:]

def mnist_loader(selected, bs=CFG.batch_size, train=True, shuffle=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("./data", train=train, download=True, transform=tfm)
    if selected is not None:
        idx = [i for i,(_,y) in enumerate(ds) if y in selected]
        sub = Subset(ds, idx)
    else:
        sub = ds
    
    return DataLoader(sub, batch_size=bs, shuffle=shuffle,
                      num_workers=CFG.num_workers, pin_memory=CFG.pin_memory)

def beta_at_epoch(ep, warmup_ep=CFG.beta_warmup_cil, beta_max=CFG.beta_max):
    if warmup_ep <= 0: return beta_max
    return beta_max * min(1.0, (ep+1)/float(warmup_ep))

def kl_with_freebits(mu, logvar, free_bits=CFG.kl_free_bits):
    kl_elem = 0.5 * (torch.exp(logvar) + mu*mu - 1.0 - logvar)
    if free_bits > 0.0:
        kl_elem = torch.clamp(kl_elem, min=free_bits)
    return kl_elem.sum(1).mean()

def vae_loss(x_logits, x, mu, lv, beta=1.0, free_bits=CFG.kl_free_bits):
    bce = F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")
    kl    = kl_with_freebits(mu, lv, free_bits)
    total = bce + beta*kl
    return total, bce, kl

def lambda_con_at_epoch(ep, warmup_ep=CFG.lambda_warmup_ep, lam_max=CFG.lambda_con_max):
    if warmup_ep <= 0: return lam_max
    return lam_max * min(1.0, (ep+1)/float(warmup_ep))

def info_nce(pos_logit, neg_logits):
    logits = torch.cat([pos_logit, neg_logits], 1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def optimize_model_for_runtime(model:nn.Module)->nn.Module:
    if DEVICE=="cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return model

@torch.no_grad()
def _to_3ch_and_resize(x: torch.Tensor, size: int):
    if x.shape[1] == 3:
        if x.shape[2] == size and x.shape[3] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x3 = x.repeat(1,3,1,1)
    return F.interpolate(x3, size=(size, size), mode="bilinear", align_corners=False)

# --- File I/O Utilities ---

def _now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def init_run_dir(label: str) -> Path:
    root = Path(CFG.out_root); root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{_now_str()}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_config(run_dir: Path, base_classes, stream_classes):
    cfg_dict = {k: getattr(CFG, k) for k in CFG.__dataclass_fields__.keys()}
    payload = {"cfg": cfg_dict, "device": DEVICE,
               "base_classes": base_classes, "stream_classes": stream_classes}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_metrics_csv(run_dir: Path, metrics: dict, filename: str):
    keys = ["steps", "global_batch"] + [k for k in metrics.keys() if k not in ["steps", "global_batch"]]
    lines = [",".join(keys)]
    n = len(metrics["steps"])
    for i in range(n):
        row = []
        for k in keys:
            vlist = metrics.get(k, [])
            val = vlist[i] if i < len(vlist) else ""
            if isinstance(val, dict):
                val = json.dumps(val)
            row.append(str(val))
        lines.append(",".join(row))
    (run_dir / filename).write_text("\n".join(lines), encoding="utf-8")

def save_json(run_dir: Path, obj: dict, filename: str):
    with open(run_dir / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_plot(fig_path: Path):
    plt.tight_layout(); plt.savefig(fig_path, dpi=150)
    if CFG.show_plots: plt.show()
    else: plt.close()

def save_image_grid(tensor_BCHW: torch.Tensor, path: Path, nrow: int = CFG.sample_grid_nrow):
    grid = make_grid(tensor_BCHW.clamp(0,1).cpu(), nrow=nrow, padding=2)
    save_image(grid, path)
    
def plot_metric(mA, mB, key, title, ylabel, save_dir: Path = None, fname: str = None):
    x_key = "global_batch" if "global_batch" in mA and "global_batch" in mB else "steps"
    
    key_a = key if key in mA else f"{key}_all"
    key_b = key if key in mB else f"{key}_all"

    plt.figure()
    if key_a in mA:
        plt.plot(mA.get(x_key, []), mA.get(key_a, []), '-o', label=f"Baseline (A) {key_a}")
    if key_b in mB:
        plt.plot(mB.get(x_key, []), mB.get(key_b, []), '-o', label=f"Contrastive (B) {key_b}")
    
    plt.xlabel("Global Batch Step"); plt.ylabel(ylabel) 
    plt.title(title); plt.grid(True); plt.legend()
    
    if CFG.save_plots and save_dir is not None and fname is not None:
        save_plot(save_dir / fname)
    else:
        if CFG.show_plots: plt.show()
        else: plt.close()

# ============================ 5. Evaluation Functions (Moved) ============================
# (v8.2: MetricCalculator 클래스 위로 이동)

@torch.no_grad()
def generate_eval_samples(model, classes, per_class: int, mu_proto_bank: dict):
    """평가를 위한 실제 이미지(X_real)와 생성 이미지(X_fake)를 반환"""
    # 1. Real 이미지 로드
    real_loader = mnist_loader(classes, bs=256, train=False, shuffle=True)
    real = []
    for x,_ in real_loader:
        real.append(x)
        if sum(t.size(0) for t in real) >= per_class*len(classes): break
    X_real = torch.cat(real,0)[:per_class*len(classes)].to(DEVICE)
    
    # 2. Fake 이미지 생성 (Generative Replay 방식)
    xs=[]
    for c in classes:
        if c not in mu_proto_bank: continue # 프로토타입이 없으면 스킵
        mu_c = mu_proto_bank[c].to(DEVICE)
        z = mu_c.unsqueeze(0) + CFG.replay_sigma*torch.randn(per_class, mu_c.numel(), device=DEVICE)
        xs.append(torch.sigmoid(model.decode(z)))
    X_fake = torch.cat(xs,0) if xs else torch.empty(0,1,28,28, device=DEVICE)
    return X_real, X_fake

@torch.no_grad()
def analyze_proto_separation(classes, proto_bank: dict):
    """프로토타입 뱅크 내의 클래스 간 분리도(Cosine/Euclidean)를 측정"""
    vecs = [F.normalize(proto_bank[c].to(DEVICE), dim=0) for c in classes if c in proto_bank]
    if len(vecs) < 2: return {}
    
    V = torch.stack(vecs,0)
    cos_sim = V @ V.t()
    eu_dist = torch.cdist(V, V, p=2)
    
    mask = ~torch.eye(V.size(0), dtype=torch.bool, device=V.device) # 대각선(자기 자신) 제외
    cos_off_diag = cos_sim[mask]
    eu_off_diag = eu_dist[mask]
    
    return {
        "cos_mean": float(cos_off_diag.mean().item()),
        "cos_min": float(cos_off_diag.min().item()),
        "eu_mean": float(eu_off_diag.mean().item()),
        "eu_min": float(eu_off_diag.min().item())
    }

# ============================ 6. Base Trainer ============================
def train_base(model, loader):
    """초기 Base 클래스들로 VAE 모델을 사전 학습시킵니다."""
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    model.train()
    
    for ep in range(CFG.epochs_base):
        beta = beta_at_epoch(ep, CFG.beta_warmup_base, CFG.beta_max)
        m_total=m_bce=m_kl=0.0
        
        for x, _ in tqdm(loader, desc=f"[BASE] {ep+1}/{CFG.epochs_base}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            if DEVICE=="cuda": x = x.to(memory_format=torch.channels_last)
            
            logits, mu, lv, _ = model(x)
            total, bce, kl = vae_loss(logits, x, mu, lv, beta, CFG.kl_free_bits)
            
            opt.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            m_total+=total.item(); m_bce+=bce.item(); m_kl+=kl.item()
        
        n = len(loader)
        print(f"[BASE] ep {ep+1} | beta={beta:.2f} | total={m_total/n:.4f} | BCE={m_bce/n:.4f} | KL={m_kl/n:.4f}")
    return model

# ============================ 7. Metric Calculator Class ============================
# [피드백 2 & 3] 평가 로직을 담당하는 별도 클래스

class MetricCalculator:
    """
    모든 평가 지표(Recon, Gen)의 계산을 담당하는 클래스.
    CIL_Trainer로부터 모델과 프로토타입 뱅크를 주입받아 사용합니다.
    """
    def __init__(self, model: VAE, classes_to_eval: list, mu_proto_bank: dict, proto_bank: dict):
        self.model = model
        self.classes_to_eval = classes_to_eval
        self.mu_proto_bank = mu_proto_bank   # For Generation
        self.proto_bank = proto_bank         # For Separation analysis
        self.metrics = {} # 계산된 지표를 누적할 딕셔너리
        
        # [v8.3] OOM 방지를 위해, Global FID/IS 계산기는 클래스 레벨에서 *재사용*
        # Per-class 계산기는 메서드 내에서 일회용으로 생성
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(DEVICE)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=CFG.lpips_backbone, reduction=None).to(DEVICE)
        
        self.global_fid = FrechetInceptionDistance().to(DEVICE)
        self.global_isc = InceptionScore().to(DEVICE)
        # Per-class 계산기는 메서드 내에서 생성 (매번 리셋)
        self.per_class_fid = FrechetInceptionDistance().to(DEVICE)
        self.per_class_isc = InceptionScore().to(DEVICE)


    def _log(self, name: str, value: float):
        """계산된 지표를 내부 딕셔너리에 저장"""
        self.metrics[name] = value

    @torch.no_grad()
    def run_all_metrics(self) -> dict:
        """
        [피드백 3]
        모든 평가(Recon, Gen, Separation)를 실행하고 W&B 로깅을 위한 dict를 반환합니다.
        """
        print(f"[{self.model.label}] Running evaluations for classes: {self.classes_to_eval}")
        self.model.eval()
        
        # [피드백 3.2 tag="Recon"] 재구성 품질 평가 (SSIM, LPIPS)
        self._calculate_reconstruction_metrics()
        
        # [피드백 3.2 tag="Gen"] 생성 품질 평가 (FID, IS)
        self._calculate_generation_metrics()
        
        # 프로토타입 분리도 평가
        self._calculate_separation_metrics()
        
        self.model.train()
        return self.metrics

    @torch.no_grad()
    def _calculate_reconstruction_metrics(self):
        """
        [피드백 2] (Recon)
        전체 테스트셋을 루프 돌며 클래스별 SSIM, LPIPS를 계산합니다.
        (이 함수는 메모리 집약적이지 않으므로 v8.2와 동일)
        """
        print("... calculating Reconstruction metrics (SSIM, LPIPS) ...")
        test_loader = mnist_loader(selected=None, bs=CFG.metric_batch, train=False, shuffle=False)
        
        all_ssim = []
        all_lpips = []
        all_y = []
        
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            mask = torch.tensor([item in self.classes_to_eval for item in y], dtype=torch.bool).to(DEVICE)
            if not mask.any():
                continue
            
            x_filtered, y_filtered = x[mask], y[mask]
            
            x_logits, _, _, _ = self.model(x_filtered)
            x_recon = torch.sigmoid(x_logits)
            
            ssim_vals = self.ssim_metric(x_recon, x_filtered)
            
            x_recon_lpips = (_to_3ch_and_resize(x_recon, CFG.lpips_resize) * 2.0) - 1.0
            x_filtered_lpips = (_to_3ch_and_resize(x_filtered, CFG.lpips_resize) * 2.0) - 1.0
            lpips_vals = self.lpips_metric(x_recon_lpips, x_filtered_lpips)
            
            all_ssim.append(ssim_vals.cpu())
            all_lpips.append(lpips_vals.cpu())
            all_y.append(y_filtered.cpu())

        if not all_y:
            print("Warning: No test data found for reconstruction metrics.")
            return

        all_ssim = torch.cat(all_ssim)
        all_lpips = torch.cat(all_lpips)
        all_y = torch.cat(all_y)
        
        for c in self.classes_to_eval:
            mask = (all_y == c)
            if mask.any():
                c_ssim = all_ssim[mask].mean().item()
                c_lpips = all_lpips[mask].mean().item()
                self._log(f"Eval/Recon/cl/{c}/SSIM", c_ssim)
                self._log(f"Eval/Recon/cl/{c}/LPIPS", c_lpips)
        
        self._log(f"Eval/Recon/all/SSIM", all_ssim.mean().item())
        self._log(f"Eval/Recon/all/LPIPS", all_lpips.mean().item())

    @torch.no_grad()
    def _calculate_generation_metrics(self):
        """
        [v8.3 OOM FIX]
        Global FID/IS 계산 시 torch.cat을 사용하지 않고, 
        per-class 루프 내에서 global metric 객체를 청크(chunk) 단위로 .update() 합니다.
        """
        print("... calculating Generation metrics (FID, IS) ...")
        
        # [v8.3] Global 계산기 리셋
        self.global_fid.reset()
        self.global_isc.reset()
        bs = CFG.metric_batch
        
        # [피드백 2.2] 클래스별 FID/IS 계산
        for c in self.classes_to_eval:
            # [피드백 1.1, 1.2] N개 샘플을 매번 새로 생성
            real_c, fake_c = self._generate_samples_for_class(c, CFG.eval_n_per_class)
            
            if real_c is None or fake_c is None:
                continue

            # 3채널, 299x299, uint8로 변환
            real_c_u8 = (_to_3ch_and_resize(real_c, CFG.eval_resize) * 255).clamp(0,255).to(torch.uint8)
            fake_c_u8 = (_to_3ch_and_resize(fake_c, CFG.eval_resize) * 255).clamp(0,255).to(torch.uint8)
            
            # [v8.3] 1. Per-Class FID (일회용 계산기 사용)
            self.per_class_fid.reset()
            for i in range(0, real_c_u8.size(0), bs): self.per_class_fid.update(real_c_u8[i:i+bs], real=True)
            for i in range(0, fake_c_u8.size(0), bs): self.per_class_fid.update(fake_c_u8[i:i+bs], real=False)
            fid_val_c = self.per_class_fid.compute().item()
            self._log(f"Eval/Gen/cl/{c}/FID", fid_val_c)
            
            # [v8.3] 2. Per-Class IS (일회용 계산기 사용)
            self.per_class_isc.reset()
            for i in range(0, fake_c_u8.size(0), bs): self.per_class_isc.update(fake_c_u8[i:i+bs])
            is_val_c, _ = self.per_class_isc.compute()
            self._log(f"Eval/Gen/cl/{c}/IS", is_val_c.item())

            # [v8.3] 3. Global FID/IS에 청크(클래스) 단위로 데이터 추가 (OOM 방지)
            for i in range(0, real_c_u8.size(0), bs): self.global_fid.update(real_c_u8[i:i+bs], real=True)
            for i in range(0, fake_c_u8.size(0), bs): self.global_fid.update(fake_c_u8[i:i+bs], real=False)
            for i in range(0, fake_c_u8.size(0), bs): self.global_isc.update(fake_c_u8[i:i+bs])

        # [v8.3] 4. 루프가 끝난 후, Global FID/IS 최종 계산
        try:
            fid_val_all = self.global_fid.compute().item()
            is_val_all, _ = self.global_isc.compute()
            self._log(f"Eval/Gen/all/FID", fid_val_all)
            self._log(f"Eval/Gen/all/IS", is_val_all.item())
        except Exception as e:
            print(f"Warning: Could not compute global FID/IS. Error: {e}")
            self._log(f"Eval/Gen/all/FID", -1.0) # 에러 발생 시 -1
            self._log(f"Eval/Gen/all/IS", -1.0)

    @torch.no_grad()
    def _generate_samples_for_class(self, c: int, n: int):
        """특정 클래스 'c'의 실제/생성 샘플 'n'개를 반환"""
        # Real (Testset에서 n개)
        real_loader_c = mnist_loader(selected=[c], bs=n, train=False, shuffle=True)
        try:
            X_real_c = next(iter(real_loader_c))[0].to(DEVICE)
        except StopIteration:
            print(f"Warning: No real test data for class {c}.")
            return None, None
        
        # Fake (mu_proto에서 n개 생성)
        if c not in self.mu_proto_bank:
            print(f"Warning: No mu-prototype for class {c}, cannot generate.")
            return X_real_c, None
        
        mu_c = self.mu_proto_bank[c].to(DEVICE)
        z = mu_c.unsqueeze(0) + CFG.replay_sigma*torch.randn(n, mu_c.numel(), device=DEVICE)
        X_fake_c = torch.sigmoid(self.model.decode(z))
        
        return X_real_c, X_fake_c

    @torch.no_grad()
    def _calculate_separation_metrics(self):
        """프로토타입 분리도(Separation)를 계산"""
        print("... calculating Separation metrics ...")
        sep_metrics = analyze_proto_separation(self.classes_to_eval, self.proto_bank)
        for k,v in sep_metrics.items():
            self._log(f"Eval/Sep/{k}", v)

# ============================ 8. CIL Trainer Class ============================
class CIL_Trainer:
    """
    CIL의 전체 학습/평가 과정을 관리하는 클래스.
    v8: 평가 로직을 MetricCalculator 클래스로 완전 분리.
    """
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
        
        print(f"[{self.label}] Initializing prototype banks for base classes...")
        self.proto_bank = self._update_proto({}, self.seen)
        self.mu_proto_bank = self._update_mu_proto({}, self.seen)

    # --- Prototype Bank Management ---
    @torch.no_grad()
    def _update_proto(self, proto_bank, classes, ema=0.9):
        """Contrastive Loss용 프로토타입 뱅크(proj(mu))를 업데이트합니다."""
        if not classes: return proto_bank
        loader = mnist_loader(classes, bs=256, train=False, shuffle=False)
        sums, counts = {c:None for c in classes}, {c:0 for c in classes}
        
        for x,y in loader:
            x = x.to(DEVICE, non_blocking=True)
            z = self.model.embed_mu(x, project=True) # proj(mu)
            for c in classes:
                m = (y==c)
                if m.any():
                    add = z[m.to(DEVICE)].sum(0)
                    sums[c] = add if sums[c] is None else sums[c]+add
                    counts[c]+= int(m.sum().item())
        
        for c in classes:
            if counts[c]==0: continue
            new = sums[c]/counts[c]
            proto_bank[c] = (ema*proto_bank[c] + (1-ema)*new) if c in proto_bank else new.clone()
        return proto_bank

    @torch.no_grad()
    def _update_mu_proto(self, mu_proto_bank, classes, ema=0.9):
        """Generative Replay용 프로토타입 뱅크(raw mu)를 업데이트합니다."""
        if not classes: return mu_proto_bank
        loader = mnist_loader(classes, bs=256, train=False, shuffle=False)
        sums, counts = {c:None for c in classes}, {c:0 for c in classes}
        
        for x,y in loader:
            x = x.to(DEVICE, non_blocking=True)
            mu,_ = self.model.encode(x) # raw mu
            for c in classes:
                m = (y==c)
                if m.any():
                    add = mu[m.to(DEVICE)].sum(0)
                    sums[c] = add if sums[c] is None else sums[c]+add
                    counts[c]+= int(m.sum().item())
        
        for c in classes:
            if counts[c]==0: continue
            new = sums[c]/counts[c]
            mu_proto_bank[c] = (ema*mu_proto_bank[c] + (1-ema)*new) if c in mu_proto_bank else new.clone()
        return mu_proto_bank

    # --- Training Helper Methods ---
    @torch.no_grad()
    def _generate_replay(self, gen_model, classes, total_needed:int):
        """Generative Replay를 위한 과거 클래스 샘플을 생성합니다."""
        if total_needed <= 0 or not classes: return None, None
        xs, ys = [], []
        per = max(CFG.min_gen_per_cls, total_needed // max(1,len(classes)))
        
        for c in classes:
            if c not in self.mu_proto_bank: continue # raw mu 프로토타입 사용
            mu_c = self.mu_proto_bank[c].to(DEVICE)
            z = mu_c.unsqueeze(0) + CFG.replay_sigma*torch.randn(per, mu_c.numel(), device=DEVICE)
            xlogits = gen_model.decode(z)
            x = torch.sigmoid(xlogits).detach()
            y = torch.full((x.size(0),), c, dtype=torch.long, device=DEVICE)
            xs.append(x); ys.append(y)
            
        if not xs: return None, None
        X, Y = torch.cat(xs,0), torch.cat(ys,0)
        
        if X.size(0) > total_needed:
            idx = torch.randperm(X.size(0), device=DEVICE)[:total_needed]
            X, Y = X[idx], Y[idx]
        return X, Y

    @torch.no_grad()
    def _build_pos_neg(self, z_new, y, seen, hard_k):
        """Contrastive Loss를 위한 Positive/Negative 벡터를 구축합니다."""
        B, D = z_new.shape
        z_pos = torch.empty_like(z_new)
        
        for c in y.unique():
            mask = (y==c)
            cls_vecs = z_new[mask]
            if cls_vecs.size(0) >= 2:
                m = cls_vecs.mean(0)
            elif int(c.item()) in self.proto_bank: # proj(mu) 프로토타입 사용
                m = self.proto_bank[int(c.item())].to(z_new.device)
            else:
                m = cls_vecs[0]
            z_pos[mask] = m

        if any(c in self.proto_bank for c in seen):
            neg_mat = torch.stack([self.proto_bank[c] for c in seen if c in self.proto_bank], dim=0).to(z_new.device)
        else:
            neg_mat = torch.zeros(1,D,device=z_new.device) # Fallback

        bmean = F.normalize(z_new.mean(0,keepdim=True),dim=1) # 배치 평균
        negn  = F.normalize(neg_mat,dim=1)
        sims  = F.cosine_similarity(bmean, negn) 
        k     = min(hard_k, neg_mat.size(0))
        _,idx = sims.topk(k)
        z_neg = neg_mat[idx]
        return z_pos, z_neg
    
    def _calculate_batch_loss(self, logits, x_all, y_all, mu, lv, beta, lam):
        """한 배치의 VAE 손실과 Contrastive 손실을 모두 계산합니다."""
        # 1. VAE Loss
        total, bce, kl = vae_loss(logits, x_all, mu, lv, beta, CFG.kl_free_bits)
        batch_p_pos = 0.0

        # 2. Contrastive Loss (B 모델이고, 과거 클래스가 있을 경우)
        if self.use_contrastive and self.seen:
            z = F.normalize(self.model.proj(mu), dim=1)
            z_pos, z_neg = self._build_pos_neg(z, y_all, self.seen, CFG.hard_k)
            z_pos = F.normalize(z_pos, dim=1); z_neg = F.normalize(z_neg, dim=1)
            
            pos_logit = (z*z_pos).sum(1, keepdim=True) / CFG.tau
            neg_logits = (z @ z_neg.t()) / CFG.tau if z_neg.numel() else torch.zeros(z.size(0),1,device=z.device)
            con_loss = info_nce(pos_logit, neg_logits)
            
            total = total + lam * con_loss
            
            batch_p_pos = float(torch.softmax(torch.cat([pos_logit, neg_logits],1),1)[:,0].mean().item())
        
        return total, bce, kl, batch_p_pos

    # --- Logging & Evaluation Helper Methods ---
    def _log_batch_to_wandb(self, total, bce, kl, batch_p_pos):
        """매 배치(train) 손실을 W&B에 로깅합니다."""
        wb_payload = {
            "train/total": total.item(),
            "train/BCE": bce.item(),
            "train/KL": kl.item(),
        }
        if self.use_contrastive:
            wb_payload["train/p_pos"] = batch_p_pos
        self.wb_run.log(wb_payload, step=self.global_batch_counter)

    # [v8.1 BUG FIX]
    def _run_epoch_evaluation(self, classes_to_eval) -> dict:
        """
        매 에폭 종료 시, MetricCalculator를 호출하여 모든 평가를 수행.
        """
        # [FIX] 계산기 생성 *전에* 트레이너의 뱅크를 먼저 업데이트합니다.
        self.proto_bank = self._update_proto(self.proto_bank, classes_to_eval, ema=0.9)
        self.mu_proto_bank = self._update_mu_proto(self.mu_proto_bank, classes_to_eval, ema=0.9)
            
        calculator = MetricCalculator(
            model=self.model,
            classes_to_eval=classes_to_eval,
            mu_proto_bank=self.mu_proto_bank, # 업데이트된 뱅크 전달
            proto_bank=self.proto_bank      # 업데이트된 뱅크 전달
        )
        all_metrics = calculator.run_all_metrics()
        return all_metrics

    def _log_epoch_to_wandb(self, epoch_metrics: dict, eval_metrics: dict):
        """매 에폭 종료 지표(학습 요약 + 평가)를 W&B에 로깅합니다."""
        
        # 1. 학습 요약 지표 (Summary)
        wb_payload = {
            "summary/total": epoch_metrics["total"],
            "summary/BCE":   epoch_metrics["bce"],
            "summary/KL":    epoch_metrics["kl"],
        }
        if self.use_contrastive:
            wb_payload["summary/p_pos"] = epoch_metrics["p_pos"]
        
        # 2. 평가 지표 (Eval)
        wb_payload.update(eval_metrics)
        
        self.wb_run.log(wb_payload, step=self.global_batch_counter)

        # 3. W&B에 샘플 이미지 로깅
        if CFG.save_sample_grid:
            with torch.no_grad():
                classes_to_sample = self.seen + [self.current_new_c]
                X_real_small, X_fake_small = generate_eval_samples(
                    self.model, classes_to_sample, per_class=8, mu_proto_bank=self.mu_proto_bank
                )
                logs = {}
                if X_real_small.numel() > 0:
                    grid_r = make_grid(X_real_small[:min(64, X_real_small.size(0))].clamp(0,1).cpu(), nrow=8, padding=2)
                    logs["eval/samples/real_grid"] = wandb.Image(grid_r)
                if X_fake_small.numel() > 0:
                    grid_f = make_grid(X_fake_small[:min(64, X_fake_small.size(0))].clamp(0,1).cpu(), nrow=8, padding=2)
                    logs["eval/samples/fake_grid"] = wandb.Image(grid_f)
                if logs:
                    self.wb_run.log(logs, step=self.global_batch_counter)

    def _save_local_artifacts(self, step, eval_metrics: dict):
        """CIL 스텝 종료 시 로컬에 로그 및 샘플 이미지를 저장합니다."""
        if self.run_dir is None:
            return
            
        step_log = {"step": step, "label": self.label, **eval_metrics} 
        with open(self.run_dir / "gen_metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(step_log) + "\n")
            
        if CFG.save_sample_grid:
            X_real_small, X_fake_small = generate_eval_samples(
                self.model, self.seen, per_class=16, mu_proto_bank=self.mu_proto_bank
            )
            save_image_grid(X_real_small[:CFG.sample_grid_nrow*CFG.sample_grid_nrow],
                            self.run_dir / f"{self.label}_step{step:02d}_real.png", nrow=CFG.sample_grid_nrow)
            save_image_grid(X_fake_small[:CFG.sample_grid_nrow*CFG.sample_grid_nrow],
                            self.run_dir / f"{self.label}_step{step:02d}_fake.png", nrow=CFG.sample_grid_nrow)

    # --- Main Training Loop ---
    def run_cil(self, stream_classes):
        """CIL 전체 학습 루프를 실행합니다."""
        
        for step, new_c in enumerate(stream_classes, 1):
            self.current_new_c = new_c
            print(f"\n[{self.label}] === Step {step} learn {new_c} ===")
            dl_new = mnist_loader([new_c], bs=CFG.batch_size, train=True, shuffle=True)
            self.model.train()

            gen_model = copy.deepcopy(self.model).eval()
            for p in gen_model.parameters(): p.requires_grad_(False)
            
            latest_eval_metrics = {} # 마지막 에폭의 평가 지표 저장용

            for ep in range(CFG.epochs_cil):
                
                beta = beta_at_epoch(ep, CFG.beta_warmup_cil, CFG.beta_max)
                lam  = lambda_con_at_epoch(ep, CFG.lambda_warmup_ep, CFG.lambda_con_max) if self.use_contrastive else 0.0

                m_total=m_bce=m_kl=m_ppos=0.0; n_batches=0

                # --- Batch Loop ---
                for x_new, y_new in dl_new:
                    x_new, y_new = x_new.to(DEVICE), y_new.to(DEVICE)
                    if DEVICE=="cuda": x_new = x_new.to(memory_format=torch.channels_last)

                    # 1. Generate Replay Data
                    need = int(x_new.size(0) * CFG.replay_ratio)
                    x_rep, y_rep = self._generate_replay(gen_model, self.seen, need)
                    
                    # 2. Concat New Data + Replay Data
                    x_all = torch.cat([x_new, x_rep], 0) if x_rep is not None else x_new
                    y_all = torch.cat([y_new, y_rep], 0) if x_rep is not None else y_new

                    # 3. Forward Pass
                    logits, mu, lv, _ = self.model(x_all)
                    
                    # 4. Calculate Loss
                    total, bce, kl, batch_p_pos = self._calculate_batch_loss(
                        logits, x_all, y_all, mu, lv, beta, lam
                    )

                    # 5. Backward Pass
                    self.opt.zero_grad(); total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()

                    # 6. Log Batch Metrics
                    m_total+=total.item(); m_bce+=bce.item(); m_kl+=kl.item()
                    m_ppos += batch_p_pos
                    n_batches += 1
                    self.global_batch_counter += 1
                    
                    if self.wb_run:
                        self._log_batch_to_wandb(total, bce, kl, batch_p_pos)
                
                # --- End of Epoch ---
                # 7. Log Epoch Metrics (Summary)
                epoch_metrics = {
                    "total": m_total/n_batches,
                    "bce": m_bce/n_batches,
                    "kl": m_kl/n_batches,
                    "p_pos": m_ppos/max(1, n_batches)
                }
                
                self.metrics["steps"].append(step) 
                self.metrics["global_batch"].append(self.global_batch_counter)
                for k,v in epoch_metrics.items(): self.metrics[k].append(v)

                # 8. Run Evaluation (Heavy)
                classes_to_eval = self.seen + [new_c]
                eval_metrics = self._run_epoch_evaluation(classes_to_eval)
                latest_eval_metrics = eval_metrics # 로컬 저장을 위해 마지막 값 저장
                
                # 9. Log Evaluation Metrics (to local dict & W&B)
                for k,v in eval_metrics.items(): 
                    self.metrics[k.replace("Eval/Gen/all/", "gen_").replace("Eval/Recon/all/", "recon_")].append(v)

                if self.wb_run:
                    self._log_epoch_to_wandb(epoch_metrics, eval_metrics)
                
                gm = eval_metrics.get("Eval/Gen/all/FID", 0.0)
                print(f"[{self.label}] step {step} ep {ep+1} | total={epoch_metrics['total']:.4f} | Global FID: {gm:.3f}")
            
            # --- End of CIL Step ---
            # 10. Update 'seen' list and save artifacts
            self.seen.append(new_c)
            self._save_local_artifacts(step, latest_eval_metrics) 
            
            # 11. Update prototype banks for *next* step's replay
            self.proto_bank = self._update_proto(self.proto_bank, self.seen)
            self.mu_proto_bank = self._update_mu_proto(self.mu_proto_bank, self.seen)
            
            print(f"[{self.label}] step {step} finished.")

        # --- End of Training ---
        if self.run_dir:
            save_metrics_csv(self.run_dir, self.metrics, f"metrics_{self.label}.csv")
            save_json(self.run_dir, self.metrics, f"metrics_{self.label}.json")
        
        return self.metrics

# ============================ 9. Main Execution ============================
def main():
    require_generation_metrics()
    set_seed(CFG.seed)
    base_classes, stream_classes = make_class_order()
    print("Device:", DEVICE)
    print(f"[Class order] base={base_classes}, stream={stream_classes}")

    # --- 로컬 저장 디렉토리 생성 ---
    runA = init_run_dir("A_baseline"); runB = init_run_dir("B_contrastive")
    save_config(runA, base_classes, stream_classes)
    save_config(runB, base_classes, stream_classes)

    # --- 1. Base Model Training ---
    print("\n--- Starting Base Model Training ---")
    base_model = optimize_model_for_runtime(VAE(CFG.z_dim).to(DEVICE))
    base_loader = mnist_loader(base_classes, CFG.batch_size, True, True)
    base_model = train_base(base_model, base_loader)
    print("--- Base Model Training Finished ---")

    # --- 2. Experiment A (Baseline) ---
    print("\n--- Starting Experiment A (Baseline) ---")
    wbA = wandb.init(
        project=CFG.wandb_project,
        name="A_baseline",
        group=CFG.wandb_group,
        tags=list(CFG.wandb_tags),
        config={k: getattr(CFG, k) for k in CFG.__dataclass_fields__.keys()} | {"variant":"A_baseline"},
        reinit=True,
    )

    modelA = optimize_model_for_runtime(VAE(CFG.z_dim).to(DEVICE))
    modelA.load_state_dict(base_model.state_dict())
    
    trainerA = CIL_Trainer(
        model=modelA,
        base_classes=base_classes,
        use_contrastive=CFG.use_contrastive_baseline, # False
        label="A",
        run_dir=runA,
        wb_run=wbA
    )
    metricsA = trainerA.run_cil(stream_classes)
    wbA.finish()
    print("--- Experiment A (Baseline) Finished ---")

    # --- 3. Experiment B (Contrastive) ---
    print("\n--- Starting Experiment B (Contrastive) ---")
    wbB = wandb.init(
        project=CFG.wandb_project,
        name="B_contrastive",
        group=CFG.wandb_group,
        tags=list(CFG.wandb_tags),
        config={k: getattr(CFG, k) for k in CFG.__dataclass_fields__.keys()} | {"variant":"B_contrastive"},
        reinit=True,
    )

    modelB = optimize_model_for_runtime(VAE(CFG.z_dim).to(DEVICE))
    modelB.load_state_dict(base_model.state_dict()) # 동일한 base_model에서 시작
    
    trainerB = CIL_Trainer(
        model=modelB,
        base_classes=base_classes,
        use_contrastive=CFG.use_contrastive_exp, # True
        label="B",
        run_dir=runB,
        wb_run=wbB
    )
    metricsB = trainerB.run_cil(stream_classes)
    print("--- Experiment B (Contrastive) Finished ---")

    # --- 4. Save Local Comparison Plots ---
    if CFG.save_plots:
        print("\n--- Saving local comparison plots ---")
        plot_metric(metricsA, metricsB, "total", "Total Loss per Step", "Total", save_dir=runB, fname="plot_total.png")
        plot_metric(metricsA, metricsB, "bce",   "Recon (BCE) per Step", "BCE", save_dir=runB, fname="plot_bce.png")
        plot_metric(metricsA, metricsB, "kl",    "KL per Step", "KL", save_dir=runB, fname="plot_kl.png")
        plot_metric(metricsA, metricsB, "recon_SSIM", "Recon SSIM (All)", "SSIM", save_dir=runB, fname="plot_ssim_recon.png")
        plot_metric(metricsA, metricsB, "gen_FID", "Gen FID (All)", "FID", save_dir=runB, fname="plot_fid_gen.png")
        plot_metric(metricsA, metricsB, "gen_IS", "Gen IS (All)", "IS", save_dir=runB, fname="plot_is_gen.png")
        plot_metric(metricsA, metricsB, "recon_LPIPS", "Recon LPIPS (All)", "LPIPS", save_dir=runB, fname="plot_lpips_recon.png")

    wbB.finish()
    print("\n--- All experiments finished ---")

if __name__ == "__main__":
    mp.freeze_support()
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    
    main()