import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from datetime import timedelta

from model.inital_model import inial_model


class continued_trainer(inial_model):
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 lr: float = 1e-4,
                 batch_size: int = 16,
                 save_dir: str = 'save_model/',
                 checkpoint_path: str = None):
        # Initialize base trainer
        super().__init__(model, device=device, lr=lr, batch_size=batch_size, save_dir=save_dir)
        # Override optimizer and scaler to ensure fresh states if not loading
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler()
        self.start_epoch = 1
        # If a checkpoint is provided, load weights and optimizer state
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    # def load_checkpoint(self, checkpoint_path: str):
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
    #     # Load model weights
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     # Load optimizer & scaler if present
    #     if 'optimizer_state_dict' in checkpoint:
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     if 'scaler_state_dict' in checkpoint:
    #         self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    #     # Restore best validation loss and epoch
    #     self.best_loss = checkpoint.get('best_loss', self.best_loss)
    #     self.start_epoch = checkpoint.get('epoch', self.start_epoch)
    #     print(f"Loaded checkpoint from {checkpoint_path}, resuming at epoch {self.start_epoch}\n")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # whole check point
            state_dict = checkpoint['model_state_dict']

            # add the weight
            self.model.load_state_dict(state_dict)

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # update the best loss and epoch
            self.best_loss = checkpoint.get('best_loss', self.best_loss)
            self.start_epoch = checkpoint.get('epoch', self.start_epoch)

            print(f"Loaded full checkpoint, resuming at epoch {self.start_epoch}, best_loss={self.best_loss}\n")

        else:
            #onle load the weight
            self.model.load_state_dict(checkpoint)
            print(f"Loaded state_dict only, best_loss remains {self.best_loss}, start_epoch={self.start_epoch}\n")

    def save_checkpoint(self,
                        epoch: int,
                        val_loss: float,
                        val_acc: float) -> str:

        filename = f"checkpoint_epoch{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}.pth"
        path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss
        }, path)
        print(f"Saved checkpoint to: {path}\n")
        return path

    def resume_training(self,
                        train_loader,
                        val_loader,
                        max_epochs: int,
                        model_type: str = 'model',
                        threshold: int = 3,
                        min_delta: float = 1e-4,
                        min_lr: float = 1e-7):


        # scheduler to reduce LR on plateau
        scheduler = ReduceLROnPlateau(self.optimizer,
                                      mode='min',
                                      factor=0.1,
                                      patience=threshold,
                                      min_lr=min_lr)

        no_improve_count = 0
        for epoch in range(self.start_epoch, max_epochs + 1):
            # Training step
            train_loss, train_acc, train_f1, train_rec, train_prec = self.train_epoch(
                train_loader, epoch)
            # Validation step
            val_loss, val_acc, val_f1, val_rec, val_prec = self.validate_epoch(
                val_loader, epoch)

            # Step scheduler
            scheduler.step(val_loss)

            # Check improvement
            if (self.best_loss - val_loss) > min_delta:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc)
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epoch(s)\n")
                # Early stopping
                if no_improve_count >= threshold:
                    print(f"Early stopping after {epoch} epochs.\n")
                    break

            # Check LR floor
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr <= min_lr:
                print(f"Learning rate has reached the minimum threshold {min_lr}. Stopping training.\n")
                break
