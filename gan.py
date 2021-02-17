import torch.nn as nn
import torch.nn.functional as F
import torch


def embed(in_chan, out_chan):
    layers = nn.Sequential(
        nn.Linear(in_chan, 32), torch.nn.ReLU(),
        nn.Linear(32, 64), torch.nn.ReLU(),
        nn.Linear(64, 128), torch.nn.ReLU(),
        nn.Linear(128, out_chan), torch.nn.ReLU()
    )

    return layers


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

   
        self.embed1 = embed(1, 400)
        self.embed2 = embed(1, 400)
        self.embed3 = embed(1, 400)

   
        self.conv1 = nn.Conv2d(6, 16, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), )
        self.conv1b = nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), )
        self.conv2b = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.fc0 = nn.Linear(100, 110)
        self.fc1 = nn.Linear(110, 128)
        self.fc1b = nn.Linear(128, 100)
        self.fc2 = nn.Linear(32 * 100, 1)

    def forward(self, x, v1, v2, v3):
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)


        v1 = v1.view(-1, 1, 400, 1)
        v2 = v2.view(-1, 1, 400, 1)
        v3 = v3.view(-1, 1, 400, 1)

        x = torch.cat([x, v1, v2, v3], 1)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv1b(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2b(x)
        x = F.leaky_relu(x, 0.2)

        x = torch.squeeze(x, dim=3)

        x = self.fc0(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc1b(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(-1, 32 * 100)

        out = self.fc2(x)

        return out


class Generator(nn.Module):

    def __init__(self, z_size):
        super(Generator, self).__init__()

        self.fc00 = nn.Linear(z_size, 150, bias=False)
        self.batchnorm00 = nn.BatchNorm1d(3)
        self.embed1 = embed(1, 150)
        self.embed2 = embed(1, 150)
        self.embed3 = embed(1, 150)


        self.conv0 = nn.Conv2d(6, 6, kernel_size=(16, 1), stride=(1, 1), padding=(8, 0), )
        self.batchnorm0 = nn.BatchNorm2d(6)

        self.conv0b = nn.Conv2d(6, 6, kernel_size=(16, 1), stride=(1, 1), padding=(8, 0), )
        self.batchnorm0b = nn.BatchNorm2d(6)

        self.conv0c = nn.Conv2d(6, 3, kernel_size=(16, 1), stride=(1, 1), padding=(8, 0), )
        self.batchnorm0c = nn.BatchNorm2d(3)

        self.fc01 = nn.Linear(153, 200, bias=False)
        self.batchnorm01 = nn.BatchNorm1d(3)

        self.resizenn0 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )

        self.conv1 = nn.Conv2d(6, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv1b = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.batchnorm1b = nn.BatchNorm2d(32)

        self.resizenn1 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )

        self.conv3 = nn.Conv2d(32, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.batchnorm3 = nn.BatchNorm2d(16)

        self.conv3b = nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.batchnorm3b = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 3, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        self.tanh4 = nn.Tanh()

    def forward(self, x, v1, v2, v3):
        x = self.fc00(x)
        x = self.batchnorm00(x)
        x = torch.unsqueeze(x, 3)
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)


        v1 = v1.view(-1, 1, 150, 1)
        v2 = v2.view(-1, 1, 150, 1)
        v3 = v3.view(-1, 1, 150, 1)

        x = torch.cat([x, v1, v2, v3], 1)

        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = F.relu(x)

        x = self.conv0b(x)
        x = self.batchnorm0b(x)
        x = F.relu(x)

        x = self.conv0c(x)
        x = self.batchnorm0c(x)
        x = F.relu(x)

        x = torch.squeeze(x, 3)

        x = self.fc01(x)
        x = self.batchnorm01(x)
        x = F.relu(x)

        x = x.view(-1, 6, 100, 1)

        x = self.resizenn0(x)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = F.relu(x)
        x = self.resizenn1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = self.conv3b(x)
        x = self.batchnorm3b(x)
        x = F.relu(x)

        x = self.conv4(x)

        out = self.tanh4(x)

        return out
