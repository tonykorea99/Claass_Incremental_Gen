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

# ==========================================
# Hyperparameters & Setup
# ==========================================
hidden_dim = 200
latent_dim = 40
epochs_per_task = 5 
learning_rate = 1e-3
batch_size = 64
image_log_interval = 50 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task 정의
TASK_LIST = [
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
]

wandb.init(
    project="Class_Incremental_Cons",
    entity="hails",
    name="Multi_Task_Anchor_Model", # 이름 변경
    config={
        "method": "Generative Replay",
        "task_split": TASK_LIST,
        "latent_dim": latent_dim,
        "image_log_interval": image_log_interval,
        "terminology": "Anchor Model" # 설정 기록
    }
)

# ==========================================
# Model (ConvVAE)
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
        
        mse = F.mse_loss(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (mse + kld) / x.size(0), mse / x.size(0)

# ==========================================
# Data Preparation
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def get_dataloader(dataset, targets):
    indices = [i for i, t in enumerate(dataset.targets) if t in targets]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)

# ==========================================
# Training Loop
# ==========================================
model = ConvVAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# [변경] 기존 previous_model -> anchor_model로 변경
anchor_model = None 
global_step = 0

print(f"Start Training on {len(TASK_LIST)} Tasks with Anchor Model...")

for task_idx, current_classes in enumerate(TASK_LIST):
    print(f"\n=== Task {task_idx+1}: Classes {current_classes} ===")
    
    train_loader = get_dataloader(full_dataset, current_classes)
    
    # Training
    for epoch in range(epochs_per_task):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            current_batch_size = x.size(0)
            
            # --- Generative Replay using Anchor Model ---
            if anchor_model is not None:
                # Anchor Model(이전 모델)에게 과거 기억 생성을 요청
                z_replay = torch.randn(current_batch_size, latent_dim).to(device)
                with torch.no_grad():
                    x_replay = anchor_model.decoder(z_replay)
                x_combined = torch.cat((x, x_replay), dim=0)
            else:
                x_combined = x
            # --------------------------------------------

            optimizer.zero_grad()
            total_loss, mse_loss = model.get_loss(x_combined)
            total_loss.backward()
            optimizer.step()
            
            # Logging
            log_dict = {
                f"Task{task_idx+1}/Step_Loss": total_loss.item(),
                f"Task{task_idx+1}/Step_MSE": mse_loss.item(),
                "global_step": global_step
            }

            # 이미지 생성 로깅 (Step 단위)
            if global_step % image_log_interval == 0:
                model.eval() 
                with torch.no_grad():
                    z_sample = torch.randn(64, latent_dim).to(device)
                    x_gen = model.decoder(z_sample).cpu()
                    grid = torchvision.utils.make_grid(x_gen, nrow=8, padding=2, normalize=True)
                    log_dict[f"Task{task_idx+1}/Generations_Steps"] = wandb.Image(grid, caption=f"Task {task_idx+1}, Step {global_step}")
                model.train()

            wandb.log(log_dict, step=global_step)
            global_step += 1

    # ==========================================
    # Task 종료 후 평가 및 Anchor 업데이트
    # ==========================================
    print(f">>> End of Task {task_idx+1}. Evaluating...")
    seen_classes = [c for t in TASK_LIST[:task_idx+1] for c in t]
    test_loader = get_dataloader(test_dataset, seen_classes)
    
    model.eval()
    test_mse = 0
    with torch.no_grad():
        for x_test, _ in test_loader:
            x_test = x_test.to(device)
            _, mse = model.get_loss(x_test)
            test_mse += mse.item()
    
    avg_test_mse = test_mse / len(test_loader)
    print(f"    [Evaluation] Validation MSE on Classes {seen_classes}: {avg_test_mse:.4f}")
    
    wandb.log({
        "Validation/Cumulative_MSE": avg_test_mse,
        "task_idx": task_idx + 1
    }, step=global_step)

    # [Update Anchor Model]
    # 현재 학습된 모델을 새로운 Anchor로 설정하여 고정
    print(f">>> Updating Anchor Model with current state.")
    anchor_model = copy.deepcopy(model)
    anchor_model.eval()
    for p in anchor_model.parameters(): p.requires_grad = False

wandb.finish()