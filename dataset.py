from torch.utils.data import Dataset
import glob
import os
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import torch

class StyleTransferDataset(Dataset):
    def __init__(self, opt):
        super(StyleTransferDataset, self).__init__()
        self.root = opt.root
        self.prob = opt.p
        self.content = glob.glob(os.path.join(self.root, 'content/*'))
        self.style = glob.glob(os.path.join(self.root, "ink/*"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((opt.input_shape, opt.input_shape)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.style)

    def __getitem__(self, item):
        style = Image.open(self.style[item]).convert("RGB")
        content = Image.open(random.choice(self.content)).convert("RGB")

        p = random.random()
        if p > self.prob:
            style = style.transpose(Image.FLIP_LEFT_RIGHT)

        style = self.transform(style)
        content = self.transform(content)

        return style, content

    @staticmethod
    def collate(batch):
        styles = []
        contents = []

        for style, content in batch:
            styles.append(style[None])
            contents.append(content[None])

        return torch.cat(styles, dim=0), torch.cat(contents, dim=0)