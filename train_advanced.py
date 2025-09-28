import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from collections import defaultdict
import re
import math

BATCH_SIZE = 6
NUM_EPOCHS = 30
BASE_LR = 5e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############### Losses ###############
class FocalRegressionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target, reduction='none')

        focal_weight = (l1_loss + 1e-8) ** self.gamma
        focal_loss = self.alpha * focal_weight * l1_loss

        return focal_loss.mean()

class AdaptiveSmoothL1Loss(nn.Module):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        beta = torch.clamp(self.beta, min=0.1, max=2.0)

        loss = torch.where(
            diff < beta,
            0.5 * diff.pow(2) / beta,
            diff - 0.5 * beta
        )
        return loss.mean()

class CombinedCountingLoss(nn.Module):
    def __init__(self, mse_weight=1.0, l1_weight=0.5, focal_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.focal_weight = focal_weight
        self.focal_loss = FocalRegressionLoss()

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)

        l1_loss = F.l1_loss(pred, target)

        focal_loss = self.focal_loss(pred, target)

        total_loss = (self.mse_weight * mse_loss +
                     self.l1_weight * l1_loss +
                     self.focal_weight * focal_loss)

        return total_loss, {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'focal': focal_loss.item()
        }


############### Attention для смешивания признаков двух ракурсов ###############
class CrossViewAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f1, f2):
        batch_size = f1.size(0)

        features = torch.stack([f1, f2], dim=1)

        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.feature_dim)
        attention = self.softmax(scores)

        attended = torch.matmul(attention, V)

        fused = attended.sum(dim=1)
        return fused


############### Model ###############
class AdvancedBoxCounter(nn.Module):
    def __init__(self, num_outputs=3, backbone='efficientnet', use_attention=True):
        super().__init__()

        if backbone == 'efficientnet':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone1 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.backbone2 = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            feature_dim = 1280

            self.backbone1.classifier = nn.Identity()
            self.backbone2.classifier = nn.Identity()

        else:
            self.backbone1 = models.resnet50(pretrained=True)
            self.backbone2 = models.resnet50(pretrained=True)
            feature_dim = self.backbone1.fc.in_features
            self.backbone1.fc = nn.Identity()
            self.backbone2.fc = nn.Identity()

        self.feature_dim = feature_dim
        self.use_attention = use_attention

        if use_attention:
            self.attention = CrossViewAttention(feature_dim)
            fusion_dim = feature_dim
        else:
            fusion_dim = feature_dim * 2

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_outputs)
        )

        self.laptop_head = nn.Linear(256, 1)
        self.tablet_head = nn.Linear(256, 1) 
        self.group_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)

        if self.use_attention:
            fused = self.attention(f1, f2)
        else:
            fused = torch.cat([f1, f2], dim=1)

        features = self.regressor[:-1](fused)

        laptop_count = self.laptop_head(features)
        tablet_count = self.tablet_head(features)
        group_count = self.group_head(features)

        counts = torch.cat([laptop_count, tablet_count, group_count], dim=1)

        return F.softplus(counts)

############### Data ###############
class AdvancedPalletDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),

                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        img1 = Image.open(pair['img1']).convert('RGB')
        img2 = Image.open(pair['img2']).convert('RGB')

        x1 = self.transform(img1)
        x2 = self.transform(img2)
        y = torch.tensor(pair['target'], dtype=torch.float32)

        return x1, x2, y

############### Trainer ###############
class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.criterion = CombinedCountingLoss()

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)

        for batch_idx, (x1, x2, y) in enumerate(self.train_loader):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)

            self.optimizer.zero_grad()

            pred = self.model(x1, x2)
            loss, components = self.criterion(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}

        return avg_loss, avg_components

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x1, x2, y in self.val_loader:
                x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
                pred = self.model(x1, x2)
                loss, _ = self.criterion(pred, y)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs=NUM_EPOCHS):
        print(f"Training on {DEVICE}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            train_loss, components = self.train_epoch(epoch)
            val_loss = self.validate()

            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_advanced.pth')
                print(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (MSE: {components['mse']:.4f}, L1: {components['l1']:.4f}, Focal: {components['focal']:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        print("Training completed!")

def create_pairs_from_metadata(unified_dir, split):
    df = pd.read_csv(os.path.join(unified_dir, f'{split}_metadata.csv'))
    groups = defaultdict(list)

    for _, row in df.iterrows():
        img_path = os.path.join(unified_dir, split, row['filename'])
        if os.path.exists(img_path):
            groups[row['pallet_name']].append({
                'path': img_path,
                'laptop': row['laptop_count'],
                'tablet': row['tablet_count'], 
                'group': row['group_box_count']
            })

    pairs = []
    for pallet_name, items in groups.items():
        if len(items) >= 2:
            items = sorted(items, key=lambda x: x['path'])
            item1, item2 = items[0], items[1]

            pairs.append({
                'img1': item1['path'],
                'img2': item2['path'],
                'target': np.array([item1['laptop'], item1['tablet'], item1['group']], dtype=np.float32),
                'pallet_name': pallet_name
            })

    return pairs

def main():
    unified_dir = 'unified_dataset'

    if not os.path.exists(unified_dir):
        print("Датасет не найден. Запустите предобработку.")
        return

    train_pairs = create_pairs_from_metadata(unified_dir, 'train')
    val_pairs = create_pairs_from_metadata(unified_dir, 'val')
    test_pairs = create_pairs_from_metadata(unified_dir, 'test')

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    train_ds = AdvancedPalletDataset(train_pairs, augment=True)
    val_ds = AdvancedPalletDataset(val_pairs, augment=False)
    test_ds = AdvancedPalletDataset(test_pairs, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)


    model = AdvancedBoxCounter(backbone='efficientnet', use_attention=True)

    trainer = AdvancedTrainer(model, train_loader, val_loader, test_loader)

    trainer.train()

    model.load_state_dict(torch.load('best_model_advanced.pth'))
    evaluate_model(model, test_loader)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            pred = model(x1, x2)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    preds_rounded = np.round(preds).astype(int)
    targets_int = targets.astype(int)

    classes = ['laptop', 'tablet', 'group_box']
    results = {}

    for i, cls in enumerate(classes):
        mse = mean_squared_error(targets_int[:, i], preds_rounded[:, i])
        mae = mean_absolute_error(targets_int[:, i], preds_rounded[:, i])
        rmse = np.sqrt(mse)

        results[cls] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
        print(f'{cls}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')

    with open('results_advanced.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    main()
