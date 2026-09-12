"""Demo complete : genere un dataset de pieces geometriques, montre l'echec
du detecteur COCO tel quel, fine-tune la tete de classification, puis evalue
avec le mAP."""

import time
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchmetrics.detection import MeanAveragePrecision

from inspectline import (
    generate_dataset,
    ShapesDataset,
    collate_fn,
    build_model,
    freeze_backbone,
    count_parameters,
    SHAPE_NAMES,
)

torch.manual_seed(0)
N_TRAIN, N_VAL = 200, 40
N_EPOCHS = 3

print("Generation du dataset synthetique...")
train_raw = generate_dataset(N_TRAIN, size=160, seed=1)
val_raw = generate_dataset(N_VAL, size=160, seed=2)
print(f"train: {len(train_raw)} images, val: {len(val_raw)} images")

train_loader = torch.utils.data.DataLoader(
    ShapesDataset(train_raw), batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    ShapesDataset(val_raw), batch_size=4, shuffle=False, collate_fn=collate_fn
)

# --- Avant fine-tuning ---
model = build_model(n_classes=len(SHAPE_NAMES) + 1)
model.eval()
sample_img, _, sample_labels = train_raw[0]
with torch.no_grad():
    pred_before = model([F.to_tensor(sample_img)])[0]
print("\n=== Avant fine-tuning ===")
print("Detections (score>0.5):", (pred_before["scores"] > 0.5).sum().item())

# --- Fine-tuning ---
freeze_backbone(model)
trainable, total = count_parameters(model)
print(f"\nParametres entrainables: {trainable:,} / {total:,}")

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)

model.train()
t0 = time.time()
for epoch in range(N_EPOCHS):
    losses = []
    for images, targets in train_loader:
        loss_dict = model(list(images), list(targets))
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"epoch {epoch+1}/{N_EPOCHS} - loss moyenne: {np.mean(losses):.4f}")
print(f"temps d'entrainement: {time.time()-t0:.1f}s")

# --- Apres fine-tuning ---
model.eval()
with torch.no_grad():
    pred_after = model([F.to_tensor(sample_img)])[0]
mask = pred_after["scores"] > 0.5
print("\n=== Apres fine-tuning ===")
print("Detections (score>0.5):", mask.sum().item())
print(
    "Formes predites:",
    [SHAPE_NAMES[label - 1] for label in pred_after["labels"][mask].tolist()],
)
print("Vraies formes:", [SHAPE_NAMES[label] for label in sample_labels])

# --- mAP ---
metric = MeanAveragePrecision(backend="faster_coco_eval")
with torch.no_grad():
    for images, targets in val_loader:
        preds = model(list(images))
        metric.update(preds, list(targets))
result = metric.compute()
print("\n=== mAP sur le jeu de validation ===")
print("mAP (IoU 0.5:0.95):", round(result["map"].item(), 4))
print("mAP@0.5:", round(result["map_50"].item(), 4))
