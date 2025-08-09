#!/usr/bin/env python3
"""
Valuta il modello quick su validation set e stampa accuracy per classe.
"""

import os
import torch
import numpy as np

from models.breed_classifier import create_breed_classifier
from utils.dataloader import create_dataloaders_from_splits


def main():
    ckpt_path = "outputs/quick_splits/quick_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint non trovato: {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu")
    breed_names = ckpt.get("breed_names")
    num_classes = ckpt.get("num_classes")

    if not breed_names or not num_classes:
        print(
            "⚠️  breed_names o num_classes mancanti nel checkpoint; provo a continuare"
        )
        if not breed_names:
            breed_names = []

    # Dataloader con stesso ordine classi
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        splits_dir="data/quick_splits",
        batch_size=64,
        num_workers=0,
        image_size=(224, 224),
        augmentation_config=None,
        allowed_breeds=breed_names if breed_names else None,
    )

    # Modello
    model = create_breed_classifier(
        model_type="simple",
        num_classes=len(breed_names) if breed_names else num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Valutazione per classe su validation
    correct = np.zeros(len(breed_names), dtype=np.int64)
    total = np.zeros(len(breed_names), dtype=np.int64)

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                total[lbl] += 1
                if preds[i].item() == lbl:
                    correct[lbl] += 1

    print("\n📊 Validation per-class accuracy:")
    for i, name in enumerate(breed_names):
        if total[i] > 0:
            acc = 100.0 * correct[i] / total[i]
            print(f"   {name:20s}: {acc:5.1f}% ({correct[i]}/{total[i]})")

    overall = 100.0 * correct.sum() / total.sum()
    print(f"\n🎯 Overall Val Accuracy: {overall:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
