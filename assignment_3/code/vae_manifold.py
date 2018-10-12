import torch
import sys
from torchvision.utils import make_grid
from a3_vae_template import save, VAE, Encoder, Decoder

def generate_manifold(model, range):

    manifold = []
    normal_dist = torch.distributions.Normal(0, 1)
    for x in range:
        for y in range:
            z = normal_dist.icdf(torch.tensor([x, y]))
            manifold.append(z)

    manifold = torch.stack(manifold)
    samples = model.decoder(manifold)
    samples = samples.view(-1, 1, 28, 28)

    return samples

def main(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)

    with torch.no_grad():
        manifold = generate_manifold(model, torch.linspace(0, 1, 15))
        image = make_grid(manifold, nrow=15)
        save(image, 'test.jpg')

if __name__ == '__main__':
    model_path = sys.argv[-1]
    main(model_path)
