from VAE import VAE
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import os
import argparse
import numpy as np
from pytorch_fid import fid_score

def interpolate(model, sqrt_K=10, c_min=-0.5, c_max=1.5, path='./interpolate.pdf', noises=[]):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    if len(noises) == 4:
        noise_a, noise_b, noise_c, noise_d = noises[0], noises[1], noises[2], noises[3]
    else:
        noise_a = torch.randn(1, model.latent_dim).to(device)
        noise_b = torch.randn(1, model.latent_dim).to(device)
        noise_c = torch.randn(1, model.latent_dim).to(device)
        noise_d = torch.randn(1, model.latent_dim).to(device)
    row, col = sqrt_K, sqrt_K
    delta_x = (c_max - c_min) / row
    delta_y = (c_max - c_min) / col
    plt.tight_layout()
    model.eval()
    with torch.no_grad():
        for i in range(row):
            noise_x = noise_a + (delta_x * i + c_min) * (noise_b - noise_a)
            noise_y = noise_d + (delta_x * i + c_min) * (noise_c - noise_d)
            for j in range(col):
                index = i * col + j
                noise = noise_x + (delta_y * j + c_min) * (noise_y - noise_x)
                imgs = model.forward(z=noise).squeeze()
                plt.subplot(row, col, index+1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(imgs.cpu().numpy(), cmap='gray')
    plt.savefig(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_training_steps', default=25000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--data_dir', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    args = parser.parse_args()

    config = 'z-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.batch_size, args.num_training_steps)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    model = VAE(1, args.latent_dim).to(device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        trainer = Trainer(device, model, optimizer, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    model.restore(restore_ckpt_path)

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1)

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = model.forward(z=torch.randn(args.batch_size, model.latent_dim).to(device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1)
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.3f}".format(fid), flush=True)

    imgs = next(real_dl)[0][0:4].to(device)
    noise = torch.randn(4, args.latent_dim).to(device)
    mu, log_var = model.encoder(imgs).chunk(2, dim=-1)
    sampled_z = model.reparameterize(mu, log_var).unsqueeze(1)
    interpolate(model, sqrt_K=14, noises=sampled_z)
