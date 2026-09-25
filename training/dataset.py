import torch

from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


class SolariaDepthDataset(Dataset):

    def __init__(
        self,
        split="train",
        image_size=256
    ):

        self.dataset = load_dataset(
            "erkam/clevr-with-depth",
            split=split
        )

        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        imagem = item["image"]
        depth = item["depth"]

        if not isinstance(imagem, Image.Image):
            imagem = Image.fromarray(imagem)

        if not isinstance(depth, Image.Image):
            depth = Image.fromarray(depth)

        imagem = imagem.convert("RGB")

        depth = depth.convert("L")

        imagem = self.image_transform(
            imagem
        )

        depth = self.depth_transform(
            depth
        )

        return imagem, depth
