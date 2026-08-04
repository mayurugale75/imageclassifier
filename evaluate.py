"""Generate F1, confusion-matrix, and multiclass ROC evaluation artifacts."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from models.resnet import ResNet18LandUse
from utils.data import make_loaders, seed_everything

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", default="checkpoints/best_model.pt"); parser.add_argument("--data-dir", default="dataset"); parser.add_argument("--output-dir", default="artifacts"); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--val-fraction", type=float, default=.2); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader, classes = make_loaders(args.data_dir, args.batch_size, args.val_fraction, args.seed)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False); model = ResNet18LandUse(len(classes), pretrained=False).to(device); model.load_state_dict(state["model_state"] if isinstance(state, dict) else state); model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), 1).cpu().numpy(); y_true.extend(labels.numpy()); y_pred.extend(probabilities.argmax(1)); y_score.extend(probabilities)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); macro_f1 = f1_score(y_true, y_pred, average="macro")
    (output / "metrics.txt").write_text(f"macro_f1: {macro_f1:.4f}\n\n" + classification_report(y_true, y_pred, target_names=classes))
    figure, axis = plt.subplots(figsize=(10, 8)); ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(ax=axis, xticks_rotation=45, colorbar=False); figure.tight_layout(); figure.savefig(output / "confusion_matrix.png", dpi=180); plt.close(figure)
    scores, targets = torch.tensor(y_score).numpy(), label_binarize(y_true, classes=range(len(classes)))
    figure, axis = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(classes):
        fpr, tpr, _ = roc_curve(targets[:, i], scores[:, i]); axis.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(targets[:, i], scores[:, i]):.2f})")
    axis.plot([0,1], [0,1], "k--"); axis.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves"); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(output / "roc_curves.png", dpi=180)
    print(f"Macro F1: {macro_f1:.4f}; artifacts written to {output}")
if __name__ == "__main__": main()
