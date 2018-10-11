import torch
from torchvision.utils import make_grid
from a3_vae_template import generate_manifold, save, VAE, Encoder, Decoder

model = torch.load('latent_2/model_epoch_40.pt', map_location=lambda storage, loc: storage)

with torch.no_grad():
    manifold = generate_manifold(model, torch.linspace(0, 1, 15))
    image = make_grid(manifold, nrow=15)
    save(image, 'test.jpg')
