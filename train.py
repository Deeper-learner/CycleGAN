import torch
import argparse
from dataset import StyleTransferDataset
from torch.utils.data import DataLoader
from trainer import CycleGANTrainer
import warnings
warnings.filterwarnings("ignore")

def train(opt):
    device = f"cpu" if torch.cuda.is_available() else "cpu"
    train_ds = StyleTransferDataset(opt)
    train_loader = DataLoader(train_ds, batch_size=opt.bs, num_workers=0, shuffle=True, collate_fn=train_ds.collate)
    cyclegantrainer = CycleGANTrainer(opt, device, train_loader)

    if opt.pretrain:
        cyclegantrainer.load(opt.model_dir)

    for epoch in range(opt.epochs):
        cyclegantrainer.train_step(epoch)
        cyclegantrainer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CycleGAN-Pytorch")
    parser.add_argument("--root", type=str, default="./imgs/")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input_shape", type=int, default=256)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="./setting")
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--model_dir", type=str, default="")
    opt = parser.parse_args()
    train(opt)