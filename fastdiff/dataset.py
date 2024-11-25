#
import PIL
import torch
from pathlib import Path
from torchvision import transforms

import os

__all__ = [
    'AFHQDataset',
]

#======================================================================#
class AFHQDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, image_size, data_class, exts=["jpg", "jpeg", "png"]
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        if data_class == "all" or data_class == None:
            self.paths = [
                p for ext in exts for p in Path(f"{root}").glob(f"**/*.{ext}")
            ]
        else:
            self.paths = [
                p for ext in exts
                for p in Path(f"{root}/{data_class}").glob(f"*.{ext}")
            ]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)

#======================================================================#
#
