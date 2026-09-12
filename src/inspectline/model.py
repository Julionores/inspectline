"""Construction et adaptation d'un Faster R-CNN allege (backbone MobileNetV3)
pre-entraine sur COCO, pour la detection de pieces geometriques."""

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(n_classes, pretrained=True):
    """Charge le detecteur pre-entraine COCO et remplace sa tete de
    classification pour n_classes categories (fond inclus)."""
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    return model


def freeze_backbone(model):
    """Gele les parametres du backbone (aucune mise a jour pendant
    l'entrainement) ; seule la tete de classification/regression reste
    entrainable."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


def count_parameters(model):
    """Retourne (parametres_entrainables, parametres_totaux)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
