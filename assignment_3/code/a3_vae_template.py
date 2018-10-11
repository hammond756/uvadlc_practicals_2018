import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import os
matplotlib.use('Agg') # save images without DISPLAY variable (on LISA)
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

ARGS = {}

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.mean_net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.Sigmoid()
        )
        self.std_net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = self.mean_net(input), self.std_net(input)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.decoder_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decoder_net(input)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cpu'):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

        self.device = device
        self.to(device)

    def reparameterize(self, mean, std):

        # TODO: check if randn_like behaves differently than MultivariateNormal.sample
        eps = torch.randn((1, self.z_dim)).to(self.device)
        return mean + torch.mul(std, eps)

    def recon_loss(self, input, output):

        stb = 1e-6
        pos = torch.mul(input, torch.log(output + stb))
        neg = torch.mul(1 - input, torch.log(1 - output + stb))

        return -torch.sum(pos + neg, dim=1)

    def reg_loss(self, mu, std):
        return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=1)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # transform input into matrix of vectors [Batch, 784]
        input = input.view(-1, 784).to(self.device)

        # full pass
        mean, std = self.encoder(input)
        z = self.reparameterize(mean, std)
        output = self.decoder(z)

        # calculate each part of the loss
        reg_loss = self.reg_loss(mean, std)
        recon_loss = self.recon_loss(input, output)

        # avarage over batch
        average_negative_elbo = torch.mean(reg_loss + recon_loss, dim=0)
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        eps = torch.randn((n_samples, self.z_dim)).to(self.device)
        sampled_ims = self.decoder(eps)
        im_means = sampled_ims.mean(dim=0)

        shape = (-1, 1, 28, 28)
        sampled_ims, im_means = sampled_ims.view(shape), im_means.view(shape)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    avg_epoch_elbo = 0

    for n, batch in enumerate(data):

        avg_elbo = model(batch)

        model.zero_grad()
        avg_elbo.backward()
        optimizer.step()

        avg_epoch_elbo = avg_epoch_elbo + (avg_elbo.item() - avg_epoch_elbo) / (n + 1)

    return avg_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save(img, path):
    npimg = img.cpu().numpy()
    plt.imsave(path, npimg.transpose(1,2,0))

def generate_manifold(model, range):

    manifold = []
    for x in range:
        for y in range:
            manifold.append(torch.tensor([x,y]))

    manifold = torch.stack(manifold)
    samples = model.decoder(manifold)
    samples = samples.view(-1, 1, 28, 28)

    return samples

def main(config):

    # create directory for output files (images, graphs and model)
    if not os.path.exists(ARGS.output_dir):
        os.makedirs(ARGS.output_dir)

    # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"RUNNING ON {device}")

    # load data and configure model
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=config.zdim, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(config.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)

        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        with torch.no_grad():
            samples, im_means = model.sample(9)
            samples = make_grid(samples, nrow=3)
            save(samples, path=f"{ARGS.output_dir}/samples_epoch_{epoch}_train_{train_elbo:.2f}_val_{val_elbo:.2f}.jpg")

        torch.save(model, f'{ARGS.output_dir}/model_epoch_{ARGS.epochs}.pt')

        break

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    #
    #  NB: I implemented this in a seperate script, see manifold.py
    # --------------------------------------------------------------------


    save_elbo_plot(train_curve, val_curve, f'{ARGS.output_dir}/elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='(Relative) path of output directory. Will be created of it doens\'t exist')

    ARGS = parser.parse_args()

    main(ARGS)
