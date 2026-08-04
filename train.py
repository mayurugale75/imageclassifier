"""Fine-tune a ResNet-18 model on EuroSAT."""
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from tqdm import tqdm
from models.resnet import ResNet18LandUse
from utils.data import make_loaders, seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset"); parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--val-fraction", type=float, default=.2); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(args.data_dir, args.batch_size, args.val_fraction, args.seed)
    model = ResNet18LandUse(len(classes), pretrained=True).to(device)
    optimizer, criterion = optim.Adam(model.classifier.parameters(), lr=args.lr), nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True); best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); correct = total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad()
            outputs = model(images); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
        model.eval(); val_correct = val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predicted = model(images.to(device)).argmax(1).cpu()
                val_correct += (predicted == labels).sum().item(); val_total += labels.size(0)
        accuracy = val_correct / val_total
        print(f"epoch={epoch} train_accuracy={correct/total:.4f} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model_state": model.state_dict(), "classes": classes, "seed": args.seed, "val_accuracy": accuracy}, output_dir / "best_model.pt")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
if __name__ == "__main__": main()
