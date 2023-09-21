import torch
from PIL import Image
from models.Generator import Generator
import argparse
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

class Transformer(object):
    def __init__(self, opt):
        self.npath = opt.npath
        self.model = Generator()
        ckpt = torch.load(opt.model_dir)
        self.model.load_state_dict(ckpt["G1"])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((opt.input_shape, opt.input_shape)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.model.eval()

    def render(self, path):
        image = Image.open(path)
        w, h = image.size
        img = self.transform(image)[None]
        with torch.no_grad():
            output = self.model(img)[0]
        output = (0.5 * (output + 0.5))
        output = transforms.Resize((h, w))(output)
        output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))
        plt.figure()
        plt.imshow(output)
        plt.xticks([])
        plt.yticks([])
        name = os.path.join(self.npath, os.path.basename(path))
        plt.savefig(name, bbox_inches="tight", pad_inches=0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser("CycleGAN-Transformer")
    parser.add_argument("--model_dir", type=str, default="./setting/last.pth")
    parser.add_argument("--input_shape", type=int, default=256)
    parser.add_argument("--npath", type=str, default="./img")
    opt = parser.parse_args()
    transformer = Transformer(opt)
    transformer.render("./imgs/content/2013-11-10 07_43_23.jpg")