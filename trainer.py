from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.utils import weights_init_normal, set_weights_init
from torch import optim
from tqdm import tqdm
from torch import nn
from utils.buffer import ReplayBuffer
import torch
import os

class CycleGANTrainer(object):
    def __init__(self, opt, device, dataloader):
        self.opt = opt
        self.device = device
        self.dataloader = dataloader
        self.G1 = Generator().to(self.device)
        self.G2 = Generator().to(self.device)
        self.D1 = Discriminator().to(self.device)
        self.D2 = Discriminator().to(self.device)

        self.G1.apply(weights_init_normal)
        self.G2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)

        self.opt_G = optim.Adam([{"params": self.G1.parameters()},
                                 {"params": self.G2.parameters()}],
                                lr=self.opt.lr)

        self.opt_D = optim.Adam([{"params": self.D1.parameters()},
                                 {"params": self.D2.parameters()}],
                                lr=self.opt.lr)

        self.loss_MSE = nn.MSELoss()
        self.loss_MAE = nn.L1Loss()

        self.fake_content_buffer = ReplayBuffer(50)
        self.fake_style_buffer = ReplayBuffer(50)


    def train_step(self, epoch):

        self.G1.train()
        self.D1.train()
        self.G2.train()
        self.D2.train()
        with tqdm(total=len(self.dataloader), desc=f'Epoch {epoch + 1}/{self.opt.epochs}', postfix=dict, mininterval=0.3) as pbar:
            for i, batch in enumerate(self.dataloader):
                styles, contents = batch[0].to(self.device), batch[1].to(self.device)

                G1_style = self.G1(contents)
                recover_content = self.G2(G1_style)

                G2_content = self.G2(styles)
                recover_style = self.G1(G2_content)

                set_weights_init([self.D1, self.D2], False)

                self.opt_G.zero_grad()

                D2_out_fake = self.D2(G1_style)
                G1_gan_loss = self.loss_MSE(D2_out_fake, torch.ones(D2_out_fake.size()).to(self.device))

                D1_out_fake = self.D1(G2_content)
                G2_gan_loss = self.loss_MSE(D1_out_fake, torch.ones(D1_out_fake.size()).to(self.device))

                Gan_Loss = G1_gan_loss + G2_gan_loss

                Cycle_Loss = self.loss_MAE(contents, recover_content) * 10 + self.loss_MAE(styles, recover_style) * 10

                G1_identity = self.G1(styles)
                G2_identity = self.G2(contents)
                Identity_Loss = self.loss_MAE(styles, G1_identity) * 10 * 0.5 + self.loss_MAE(contents,
                                                                                              G2_identity) * 10 * 0.5

                loss_G = Gan_Loss + Cycle_Loss + Identity_Loss
                loss_G.backward()
                self.opt_G.step()

                set_weights_init([self.D1, self.D2], True)
                self.opt_D.zero_grad()
                content_fake_p = self.fake_content_buffer.query(G2_content)
                style_fake_p = self.fake_style_buffer.query(G1_style)

                D1_out_fake = self.D1(content_fake_p.detach()).squeeze()
                D1_out_real = self.D1(contents).squeeze()
                loss_D11 = self.loss_MAE(D1_out_fake, torch.zeros(D1_out_fake.size()).to(self.device))
                loss_D12 = self.loss_MAE(D1_out_real, torch.ones(D1_out_real.size()).to(self.device))
                loss_D1 = (loss_D11 + loss_D12) * 0.5

                D2_out_fake = self.D2(style_fake_p.detach()).squeeze()
                D2_out_real = self.D2(styles).squeeze()
                loss_D21 = self.loss_MAE(D2_out_fake, torch.zeros(D2_out_fake.size()).to(self.device))
                loss_D22 = self.loss_MAE(D2_out_real, torch.ones(D2_out_real.size()).to(self.device))
                loss_D2 = (loss_D21 + loss_D22) * 0.5

                loss_D = loss_D1 + loss_D2

                loss_D.backward()
                self.opt_D.step()

                pbar.update(1)
                pbar.set_postfix(**{"loss_D": loss_D.item(),
                                    "loss_G": loss_G.item()})

    def save(self):
        if not os.path.exists(self.opt.save_dir):
            os.mkdir(self.opt.save_dir)

        ckpt = {
            "G1": self.G1.state_dict(),
            "G2": self.G2.state_dict(),
            "D1": self.D1.state_dict(),
            "D2": self.D2.state_dict(),
        }
        print(f"model has been saved to {os.path.join(self.opt.save_dir, 'last.pth')}")
        torch.save(ckpt, os.path.join(self.opt.save_dir, "last.pth"))


    def load(self, model_dir):
        ckpt = torch.load(model_dir)
        self.G1.load_state_dict(ckpt["G1"])
        self.G2.load_state_dict(ckpt["G2"])
        self.D1.load_state_dict(ckpt["D1"])
        self.D2.load_state_dict(ckpt["D2"])

