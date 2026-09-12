"""Dataset PyTorch pour la detection d'objets : sert les images generees a
la demande, au format attendu par les modeles de detection torchvision."""

import torch
from torchvision.transforms import functional as F


class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        """raw_data : liste de tuples (image PIL, boxes ndarray, labels ndarray),
        typiquement produite par data.generate_dataset()."""
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        img, boxes, labels = self.raw_data[idx]
        image_tensor = F.to_tensor(img)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels + 1, dtype=torch.int64),  # +1 : 0 = fond
        }
        return image_tensor, target


def collate_fn(batch):
    """Regroupe une liste de paires (image, target) en deux listes paralleles,
    sans empilement tensoriel (images et nombre de boites de tailles variables)."""
    return tuple(zip(*batch))
