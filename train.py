import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from model import RestorationNet
from dataset import SemiconductorDataset

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run. Please run with a GPU-enabled PyTorch environment.')

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    print(f"Using device: {device}")

    model = RestorationNet(
        in_channels=1, 
        out_channels=1, 
        features=64, 
        num_rrdb=6, 
        upscale=2
    ).to(device)
    
    print(f"Model: RRDB-Net | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CharbonnierLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    use_amp = True
    scaler = GradScaler()
    
    train_dataset = SemiconductorDataset(
        args.train_degraded, 
        args.train_gt, 
        augment=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (degraded, gt) in enumerate(train_loader):
            degraded = degraded.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(degraded)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model (loss: {best_loss:.6f})")
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KLA Semiconductor Restoration Model")
    parser.add_argument('--train_degraded', type=str, required=True)
    parser.add_argument('--train_gt', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4, help='Reduce to 2 or 4 if on CPU with limited RAM')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    args = parser.parse_args()
    train(args)