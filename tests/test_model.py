import torch
from torchvision.transforms import functional as F

from inspectline import (
    generate_dataset,
    ShapesDataset,
    collate_fn,
    build_model,
    freeze_backbone,
    count_parameters,
    SHAPE_NAMES,
)


def test_build_model_has_correct_number_of_output_classes():
    model = build_model(n_classes=len(SHAPE_NAMES) + 1)
    # +1 pour la classe fond, ajoutee automatiquement par FastRCNNPredictor
    assert model.roi_heads.box_predictor.cls_score.out_features == len(SHAPE_NAMES) + 1


def test_freeze_backbone_disables_gradients_on_backbone_only():
    model = build_model(n_classes=5)
    freeze_backbone(model)

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.roi_heads.parameters())


def test_count_parameters_reflects_frozen_backbone():
    # Note : meme sans appeler freeze_backbone, torchvision gele deja les
    # parametres de BatchNorm du backbone (FrozenBatchNorm2d), une convention
    # des modeles de detection car les tailles de batch y sont trop petites
    # pour des statistiques de normalisation fiables. trainable_before est
    # donc deja legerement inferieur a total_before.
    model = build_model(n_classes=5)
    trainable_before, total_before = count_parameters(model)
    assert trainable_before <= total_before

    freeze_backbone(model)
    trainable_after, total_after = count_parameters(model)
    assert trainable_after < trainable_before
    assert total_after == total_before  # le nombre total de parametres ne change pas
    assert total_after == total_before  # le nombre total de parametres ne change pas


def test_pretrained_model_detects_nothing_on_synthetic_shapes():
    """Documente le point pedagogique central du projet : un detecteur COCO
    tel quel ne reconnait pas ces formes geometriques comme des objets."""
    model = build_model(n_classes=len(SHAPE_NAMES) + 1, pretrained=True)
    model.eval()

    raw = generate_dataset(1, size=160, seed=0)
    image, _, _ = raw[0]
    with torch.no_grad():
        pred = model([F.to_tensor(image)])[0]

    assert (pred["scores"] > 0.5).sum().item() == 0


def test_fine_tuned_model_detects_shapes_after_short_training():
    """Test d'integration plus lent (entrainement reel sur un petit jeu de
    donnees) : verifie que le fine-tuning produit un modele qui detecte
    reellement quelque chose, pas seulement que le code s'execute sans erreur."""
    torch.manual_seed(0)
    raw_train = generate_dataset(40, size=160, seed=1)

    model = build_model(n_classes=len(SHAPE_NAMES) + 1)
    freeze_backbone(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.005, momentum=0.9)

    ds = ShapesDataset(raw_train)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    model.train()
    for images, targets in loader:
        loss_dict = model(list(images), list(targets))
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    image, _, _ = raw_train[0]
    with torch.no_grad():
        pred = model([F.to_tensor(image)])[0]

    assert (pred["scores"] > 0.3).sum().item() > 0
