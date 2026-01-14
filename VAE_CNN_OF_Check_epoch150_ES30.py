import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import wandb
import copy
import numpy as np

# ==========================================
# Early Stopping (이전과 동일)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

# ==========================================
# Hyperparameters
# ==========================================
hidden_dim = 200
latent_dim = 40
epochs_per_task = 150
learning_rate = 1e-3
batch_size = 64
image_log_interval = 100
patience_limit = 20  # 테스트용 적절값
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_LIST = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

wandb.init(
    project="Class_Incremental_Cons",
    entity="hails",
    name="Detailed_Logging_Experiment",
    config={"method": "Generative Replay", "patience": patience_limit}
)

# ==========================================
# Model (ConvVAE) - Return 값 수정!
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Encoder/Decoder 정의는 동일하므로 생략 (위 코드와 동일)
        self.encoder = ConvEncoder(latent_dim) # 이전 코드 참조
        self.decoder = ConvDecoder(latent_dim) # 이전 코드 참조

    def get_loss(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_hat = self.decoder(z)
        
        # [수정] Loss를 쪼개서 리턴합니다.
        # reduction='sum' 후 batch_size로 나누는 것이 일반적입니다.
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        total_loss = recon_loss + kld_loss
        
        # x_hat(복원 이미지)도 같이 리턴해서 눈으로 확인합시다.
        return total_loss, recon_loss, kld_loss, x_hat

# (Encoder/Decoder 클래스가 없으면 에러나므로 간략히 포함)
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.flatten_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))

# ==========================================
# Main Loop
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def get_dataloader(dataset, targets, shuffle=True):
    indices = [i for i, t in enumerate(dataset.targets) if t in targets]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle)

model = ConvVAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
anchor_model = None 
global_step = 0

for task_idx, current_classes in enumerate(TASK_LIST):
    print(f"\n=== Task {task_idx+1}: Classes {current_classes} ===")
    
    train_loader = get_dataloader(full_dataset, current_classes, shuffle=True)
    val_loader = get_dataloader(val_dataset_full, current_classes, shuffle=False)
    early_stopper = EarlyStopping(patience=patience_limit, verbose=False)

    for epoch in range(epochs_per_task):
        model.train()
        # Epoch 단위 로스 집계용
        train_recon_sum = 0
        train_kld_sum = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            current_batch_size = x.size(0)
            
            # --- [Input 구성] ---
            if anchor_model is not None:
                z_replay = torch.randn(current_batch_size, latent_dim).to(device)
                with torch.no_grad():
                    x_replay = anchor_model.decoder(z_replay)
                x_combined = torch.cat((x, x_replay), dim=0) # [Real; Replay]
            else:
                x_combined = x
            # --------------------

            optimizer.zero_grad()
            # [수정] 3가지 Loss와 복원 이미지(x_hat)를 모두 받음
            total_loss, recon_loss, kld_loss, x_hat = model.get_loss(x_combined)
            total_loss.backward()
            optimizer.step()
            
            train_recon_sum += recon_loss.item()
            train_kld_sum += kld_loss.item()
            
            # =================================================
            # [시각화] Input vs Output 비교 (제일 중요!)
            # =================================================
            if global_step % image_log_interval == 0:
                with torch.no_grad():
                    # 1. Reconstruction Check: 모델이 입력을 얼마나 잘 복원했는가?
                    # x_combined: 실제 모델에 들어간 입력 (Real + Replay)
                    # x_hat: 모델이 뱉어낸 출력
                    n_show = min(16, x_combined.size(0))
                    
                    # 위쪽: 입력(Input), 아래쪽: 출력(Recon)
                    compare_img = torch.cat([x_combined[:n_show], x_hat[:n_show]])
                    grid_compare = torchvision.utils.make_grid(compare_img, nrow=n_show, padding=2, normalize=True)
                    
                    # 2. Generation Check: 모델이 꿈꾸는 이미지 (Random Noise)
                    z_sample = torch.randn(32, latent_dim).to(device)
                    x_gen = model.decoder(z_sample)
                    grid_gen = torchvision.utils.make_grid(x_gen, nrow=8, padding=2, normalize=True)

                    wandb.log({
                        f"Task{task_idx+1}/Vis_Input_vs_Recon": wandb.Image(grid_compare, caption="Top: Input (Real+Replay), Bottom: Reconstruction"),
                        f"Task{task_idx+1}/Vis_Generation": wandb.Image(grid_gen, caption="Random Generation"),
                    }, step=global_step)
            
            # Step별 Loss 기록 (Recon Loss 따로!)
            wandb.log({
                f"Task{task_idx+1}/Step_Total_Loss": total_loss.item(),
                f"Task{task_idx+1}/Step_Recon_Loss": recon_loss.item(), # 화질
                f"Task{task_idx+1}/Step_KLD_Loss": kld_loss.item()     # 분포
            }, step=global_step)

            global_step += 1

        # Validation Loop (Epoch 마다)
        model.eval()
        val_recon_sum = 0
        val_kld_sum = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                val_total, val_recon, val_kld, _ = model.get_loss(x_val)
                val_recon_sum += val_recon.item()
                val_kld_sum += val_kld.item()
        
        avg_val_loss = (val_recon_sum + val_kld_sum) / len(val_loader)
        avg_val_recon = val_recon_sum / len(val_loader)

        print(f"[Epoch {epoch+1}] Val Total: {avg_val_loss:.4f} | Val Recon: {avg_val_recon:.4f}")
        
        # [중요] Validation도 Recon Loss를 따로 봅니다.
        wandb.log({
            f"Task{task_idx+1}/Epoch_Val_Total_Loss": avg_val_loss,
            f"Task{task_idx+1}/Epoch_Val_Recon_Loss": avg_val_recon,
            "epoch": epoch
        }, step=global_step)

        # Early Stopping
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f">>> Early Stopping at Epoch {epoch+1}")
            break

    model.load_state_dict(early_stopper.best_model_state)
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters(): p.requires_grad = False

wandb.finish()