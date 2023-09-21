import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, attention=False):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(3, 64, 4, stride=2, padding=1),
                 nn.Identity(64),
                 nn.LeakyReLU(0.2, inplace=True)
                 ]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]
        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)
                  ]
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.model(x)
        return x



