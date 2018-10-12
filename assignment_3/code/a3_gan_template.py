import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.net(img)

class MiniMax(nn.Module):
    def __init__(self):
        super(MiniMax, self).__init__()

    def discriminator_loss(self, pred_real, pred_fake):
        return -0.5 * torch.log(pred_real).mean(dim=0) - 0.5 * torch.log(1 - pred_fake).mean(dim=0)

    def generator_loss(self, pred_real, pred_fake):
        return -self.discriminator_loss(pred_real, pred_fake)

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, game):

    discriminator = discriminator.to(args.device)
    generator = generator.to(args.device)

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs.to(args.device)
            imgs = imgs.view(-1, 784)

            # generate images
            z = torch.randn(args.batch_size, args.latent_dim).to(args.device)
            generated_images = generator(z)

            # score both sets of images using current discriminator parameters
            pred_real = discriminator(imgs)
            pred_fake = discriminator(generated_images)

            # calculate losses
            # loss_G = 0.5 * torch.log(pred_real).mean(dim=0) + 0.5 * torch.log(1 - pred_fake).mean(dim=0)
            loss_G = game.generator_loss(pred_real, pred_fake)
            # loss_D = -0.5 * torch.log(pred_real).mean(dim=0) - 0.5 * torch.log(1 - pred_fake).mean(dim=0)
            loss_D = game.discriminator_loss(pred_real, pred_fake)

            # backward passes
            generator.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            discriminator.zero_grad()
            loss_D.backward()
            optimizer_D.step()


            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(generated_images[:25].view(-1, 1, 28, 28),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
                pass

            if batches_done % args.print_interval == 0:
                print(f'{batches_done}/{args.n_epochs*len(dataloader)}\t\tD:{loss_D.item():.5f}\t\tG:{loss_G.item():.5f}')

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    game = MiniMax()

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, game)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator, "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='torch.device')
    parser.add_argument('--print_interval', type=int, default=20,
                        help='print every PRINT_INTERVAL iteratrions')
    args = parser.parse_args()

    main()
