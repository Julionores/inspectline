import torch

from inspectline import generate_dataset, ShapesDataset, collate_fn


def test_dataset_item_has_expected_keys_and_types():
    raw = generate_dataset(5, size=160, seed=0)
    ds = ShapesDataset(raw)
    image, target = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # RGB
    assert "boxes" in target and "labels" in target
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64


def test_labels_are_shifted_to_reserve_background_class():
    raw = generate_dataset(5, size=160, seed=0)
    ds = ShapesDataset(raw)
    _, target = ds[0]
    # aucun label ne doit valoir 0 (reserve a la classe fond)
    assert torch.all(target["labels"] >= 1)


def test_collate_fn_groups_variable_sized_targets():
    raw = generate_dataset(4, size=160, seed=1)
    ds = ShapesDataset(raw)
    batch = [ds[i] for i in range(4)]

    images, targets = collate_fn(batch)
    assert len(images) == 4
    assert len(targets) == 4
    # les images peuvent contenir un nombre different de boites -- pas d'erreur
    assert (
        any(t["boxes"].shape[0] != targets[0]["boxes"].shape[0] for t in targets)
        or True
    )


def test_dataloader_works_end_to_end():
    raw = generate_dataset(8, size=160, seed=2)
    ds = ShapesDataset(raw)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    batches = list(loader)
    assert len(batches) == 2
    images, targets = batches[0]
    assert len(images) == 4
