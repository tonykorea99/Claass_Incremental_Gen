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
# 1. Early Stopping 
# ==========================================
class EarlyStopping:
    """
    Validation Loss가 개선되지 않으면 학습을 조기 종료하고,
    '가장 좋았던 시점(Best)'의 모델로 되돌립니다.
    """
    def __init__(self, patience=10, verbose=False, delta=0):
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
            if self.verbose and (self.counter % 5 == 0 or self.counter >= self.patience):
                print(f'   [EarlyStopping] No improvement: Current({val_loss:.4f}) > Best({self.best_loss:.4f}) | Count: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'   [EarlyStopping] Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

# ==========================================
# 2. Hyperparameters & Setup
# ==========================================
hidden_dim = 200
latent_dim = 40
epochs_per_task = 150
learning_rate = 1e-3
batch_size = 64
image_log_interval = 100
patience_limit = 10  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_LIST = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

wandb.init(
    project="Class_Incremental_Cons",
    entity="hails",
    name="Multi_Task_INPUT_Check_Fixed",
    config={
        "method": "Generative Replay",
        "patience": patience_limit,
        "note": "Replaced Recon comparison with Direct Input Check (Shuffled)"
    }
)

# ==========================================
# 3. Model (ConvVAE)
# ==========================================
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

class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def get_loss(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_hat = self.decoder(z)
        
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + kld_loss
        
        return total_loss, recon_loss, kld_loss, x_hat

# ==========================================
# 4. Data & Training Loop
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

print(f"Start Training (Patience: {patience_limit})...")

for task_idx, current_classes in enumerate(TASK_LIST):
    print(f"\n=== Task {task_idx+1}: Classes {current_classes} ===")
    
    train_loader = get_dataloader(full_dataset, current_classes, shuffle=True)
    val_loader = get_dataloader(val_dataset_full, current_classes, shuffle=False)
    
    early_stopper = EarlyStopping(patience=patience_limit, verbose=True)

    for epoch in range(epochs_per_task):
        model.train()
        train_recon_sum = 0
        train_kld_sum = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            current_batch_size = x.size(0)
            
            # --- [Input Mix] ---
            # Task 1: x_combined = Real Data Only
            # Task 2+: x_combined = Real Data + Replay Data (Mixed)
            if anchor_model is not None:
                z_replay = torch.randn(current_batch_size, latent_dim).to(device)
                with torch.no_grad():
                    x_replay = anchor_model.decoder(z_replay)
                x_combined = torch.cat((x, x_replay), dim=0)
            else:
                x_combined = x
            # -------------------

            # --- [Logging: INPUT Check] ---

            if global_step % image_log_interval == 0:
                with torch.no_grad():
                    # 1. Input Check (Real + Replay가 섞여있는지 확인)
                    n_show = min(64, x_combined.size(0))
                    idx = torch.randperm(x_combined.size(0))[:n_show]
                    real_input_vis = x_combined[idx]
                    
                    grid_input = torchvision.utils.make_grid(real_input_vis, nrow=8, padding=2, normalize=True)
                    
                    # 2. Generation Test 
                    z_sample = torch.randn(64, latent_dim).to(device)
                    x_gen = model.decoder(z_sample)
                    grid_gen = torchvision.utils.make_grid(x_gen, nrow=8, padding=2, normalize=True)

                    wandb.log({
                        f"Task{task_idx+1}/INPUT_Check": wandb.Image(grid_input, caption=f"Task {task_idx+1} Input Mix (Real+Replay)"),
                        f"Task{task_idx+1}/Vis_Generation": wandb.Image(grid_gen, caption="Random Generation from Latent"),
                    }, step=global_step)

            # --- [Training Step] ---
            optimizer.zero_grad()
            total_loss, recon_loss, kld_loss, _ = model.get_loss(x_combined)
            total_loss.backward()
            optimizer.step()
            
            train_recon_sum += recon_loss.item()
            train_kld_sum += kld_loss.item()
            
            wandb.log({
                f"Task{task_idx+1}/Step_Total_Loss": total_loss.item(),
                f"Task{task_idx+1}/Step_Recon_Loss": recon_loss.item(),
                f"Task{task_idx+1}/Step_KLD_Loss": kld_loss.item()
            }, step=global_step)

            global_step += 1

        # Validation Loop
        model.eval()
        val_recon_sum = 0
        val_kld_sum = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                _, val_recon, val_kld, _ = model.get_loss(x_val)
                val_recon_sum += val_recon.item()
                val_kld_sum += val_kld.item()
        
        avg_val_loss = (val_recon_sum + val_kld_sum) / len(val_loader)
        avg_val_recon = val_recon_sum / len(val_loader)

        print(f"[Epoch {epoch+1}] Val Total: {avg_val_loss:.4f} | Val Recon: {avg_val_recon:.4f}")
        
        wandb.log({
            f"Task{task_idx+1}/Epoch_Val_Total_Loss": avg_val_loss,
            f"Task{task_idx+1}/Epoch_Val_Recon_Loss": avg_val_recon,
            "epoch": epoch
        }, step=global_step)

        # Early Stopping Check
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f">>> Early Stopping triggered at Epoch {epoch+1}")
            wandb.log({"Early_Stop_Triggered": task_idx+1}, step=global_step)
            break

    # Next Task Preparation
    print(f">>> Loading best model weights...")
    model.load_state_dict(early_stopper.best_model_state)
    
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters(): p.requires_grad = False

wandb.finish()