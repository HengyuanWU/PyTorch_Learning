# utils/trainer.py

import os
import torch

def train_epoch(model, loader, optimizer, criterion, device, log_interval):
    model.train()
    running_loss = 0.
    for idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % log_interval == 0:
            print(f"[{idx*len(x)}/{len(loader.dataset)}] loss: {running_loss/log_interval:.4f}")
            running_loss = 0.

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f"Val loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    return avg_loss, acc

def fit(model,
        train_loader, val_loader,
        optimizer, criterion,
        device,
        epochs=10,
        patience=3,
        min_delta=0.,
        log_interval=100,
        output_dir="outputs",
        prefix="model"):
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float('inf')
    no_improve = 0
    best_path = os.path.join(output_dir, f"{prefix}_best.pth")
    history = []

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch} ===")
        train_epoch(model, train_loader, optimizer, criterion, device, log_interval)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((val_loss, val_acc))

        # Early Stopping
        if best_loss - val_loss > min_delta:
            best_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(), best_path)
            print(f"  → loss improved, saved to {best_path}")
        else:
            no_improve += 1
            print(f"  → no improvement ({no_improve}/{patience})")

        # always save checkpoint
        ckpt = os.path.join(output_dir, f"{prefix}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(torch.load(best_path))
            break

    return history
